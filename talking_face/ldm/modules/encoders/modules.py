import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput
import kornia
from typing import Optional, Tuple
import numpy as np

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test

WAV2VEC_CONFIG = "facebook/wav2vec2-base-960h"


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

        
# # original embedder
# class ClassEmbedder(nn.Module):
#     def __init__(self, embed_dim, n_classes=1000, key='class_label'):
#         super().__init__()
#         self.key = key
#         self.embedding = nn.Embedding(n_classes, embed_dim)

#     def forward(self, batch, key=None):
#         if key is None:
#             key = self.key
#         # this is for use in crossattn
#         c = batch[key][:, None]
#         c = self.embedding(c)
#         return c


# trainable null embedding    
class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes, key='class_label', p_uncond=0.2):
        super().__init__()
        self.p_uncond = p_uncond
        self.n_classes = n_classes
        self.key = key
        if self.p_uncond > 0:
            self.embedding = nn.Embedding(n_classes+1, embed_dim)
        else:
            self.embedding = nn.Embedding(n_classes+1, embed_dim)
        self.total = 0
        self.drop = 0
        
    def forward(self, batch, training, key=None):
        if training:
            self.total += 1
        if key is None:
            key = self.key
        # this is for use in crossattn
        if torch.rand(1) < self.p_uncond and training:
            self.drop += 1
            b = batch[key].shape[0]
            c = torch.tensor([self.n_classes]*b)[:, None].cuda()
            c = self.embedding(c)
            print(torch.sum(self.embedding.weight.data[self.n_classes,:]))
            print(f"Running class label dropout percentage: {self.drop / self.total}")
        else:
            c = batch[key][:, None]
            c = self.embedding(c)
        return c
    

class Conv1DTemporalAttention(nn.Module):
    def __init__(self, seq_len, subspace_dim=768, subspace2hidden=False, hidden_dim=None):
        super().__init__()
        self.seq_len = seq_len
        self.subspace_dim = subspace_dim
        self.subspace2hidden = subspace2hidden
        if self.subspace2hidden:
            assert hidden_dim is not None
            self.hidden_dim = hidden_dim
            self.subspace2hidden = nn.Linear(self.subspace_dim, self.hidden_dim)
        self.attentionConvNet = nn.Sequential(
            nn.Conv1d(self.subspace_dim, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),            
            nn.Conv1d(192, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),   
            nn.Softmax(dim=1)
        )
        
    def forward(self, x): # input shape: b x seq_len x subspace_dim
        b = x.shape[0]
        x = torch.transpose(x, 1, 2) # b x subspace_dim x seq_len
        att_conv_res = self.attentionConvNet(x)
        attention = self.attentionNet(att_conv_res.view(b, self.seq_len)).view(b, self.seq_len, 1)
        result_subspace = torch.bmm(x, attention).view(b, self.subspace_dim)
        if self.subspace2hidden:
            hidden = self.subspace2hidden(result_subspace)
            return hidden
        else:
            return result_subspace.unsqueeze(1)
        
        
class Conv1DTemporalAttentionV2(nn.Module):
    def __init__(self, seq_len, subspace_dim=29, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.subspace_dim = subspace_dim
        self.hidden_dim = hidden_dim
        self.expander = nn.Sequential(
            nn.Conv1d(29, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionConvNet = nn.Sequential(
            nn.Conv1d(29, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),            
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),   
            nn.Softmax(dim=1)
        )
        
    def forward(self, x): # input shape: b x seq_len x subspace_dim
        b = x.shape[0]
        x = torch.transpose(x, 1, 2) # b x subspace_dim x seq_len
        y = self.expander(x)
        att_conv_res = self.attentionConvNet(x)
        attention = self.attentionNet(att_conv_res.view(b, self.seq_len)).view(b, self.seq_len, 1)
        result_subspace = torch.bmm(y, attention).view(b, self.hidden_dim)
        return result_subspace.unsqueeze(1)
        

# the implementation of Wav2Vec2Model is borrowed from https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
# initialize audio encoder with the pre-trained wav2vec 2.0 weights.
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask
        
        
# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)
        
        
class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, frame_num=None):
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2) # B x T x D
        
        # Output frequency of the encoder is 49 Hz according to https://arxiv.org/pdf/2006.11477.pdf
        # MEAD video are in 30 fps
        hidden_states = linear_interpolation(hidden_states, 49, 30, output_len=frame_num)
        
        if attention_mask is not None:
            print('------------- WARNING 1 -------------')
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            attention_mask = torch.zeros(
                hidden_states.shape[:2], dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[
                (torch.arange(attention_mask.shape[0], device=hidden_states.device), output_lengths - 1)
            ] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        
        hidden_states = self.feature_projection(hidden_states)[0] # return hidden_states, norm_hidden_states
        
        if self.config.apply_spec_augment and self.training:
#             print('------------- WARNING 2 -------------')
            batch_size, sequence_length, hidden_size = hidden_states.size()
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(hidden_states.dtype)
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    

class AudioEmbedder(nn.Module):
    def __init__(self, win_len=4, subspace_dim=768):
        super().__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained(WAV2VEC_CONFIG)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.win_len = win_len
        self.subspace_dim = subspace_dim
        self.attentionConvNet = nn.Sequential(
            nn.Conv1d(self.subspace_dim, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),            
            nn.Conv1d(192, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=2*self.win_len+1, out_features=2*self.win_len+1, bias=True),   
            nn.Softmax(dim=1)
        )        
        
    def forward(self, x, num_frames, frame_idx):
        assert frame_idx < num_frames
        b = x.shape[0]
        x = self.audio_encoder(input_values=x, frame_num=num_frames).last_hidden_state
        x = x[:, max(frame_idx-self.win_len, 0):min(frame_idx+self.win_len+1, num_frames), :]
        x = torch.transpose(x, 1, 2)
        if frame_idx-self.win_len < 0:
            x = F.pad(x, (-(frame_idx-self.win_len), 0), "replicate")
        elif frame_idx+self.win_len > num_frames-1:
            x = F.pad(x, (0, frame_idx+self.win_len-num_frames+1), "replicate")
        att_conv_res = self.attentionConvNet(x)
        attention = self.attentionNet(att_conv_res.view(b, 2*self.win_len+1)).view(b, 2*self.win_len+1, 1)
        result_subspace = torch.bmm(x, attention).view(b, self.subspace_dim)
        return result_subspace.unsqueeze(1)
    
    
class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=96, output_dim=128):
        super(LandmarkEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )  
        
    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(1)
        
    
class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
    

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)