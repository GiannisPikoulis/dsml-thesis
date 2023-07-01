import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import numpy as np
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

# original embedder
class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class_label'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c
    
# untrainable null embedding
class ClassEmbedder2(nn.Module):
    def __init__(self, embed_dim, n_classes, key='class_label', p_uncond=0.2):
        super().__init__()
        self.p_uncond = p_uncond
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.uncond_embedding = nn.Embedding(1, embed_dim)
        #print(self.uncond_embedding.weight.data)
        for p in self.uncond_embedding.parameters():
            p.requires_grad = False
        self.total = 0
        self.drop = 0
        
    def forward(self, batch, training, key=None):
        self.total += 1
        if key is None:
            key = self.key
        if torch.rand(1) < self.p_uncond and training:
            self.drop += 1
            b = batch[key].shape[0]
            print(self.uncond_embedding.weight.data[0,:10])
            c = torch.tensor([0]*b)[:, None].cuda()
            c = self.uncond_embedding(c)
        else:
            # this is for use in crossattn
            c = batch[key][:, None]
            print(c.view(-1))
            c = self.embedding(c)
        print(self.drop/self.total)
        return c
    

# trainable null embedding    
class ClassEmbedder3(nn.Module):
    def __init__(self, embed_dim, n_classes, key='class_label', p_uncond=0.2):
        super().__init__()
        self.p_uncond = p_uncond
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.uncond_embedding = nn.Embedding(1, embed_dim)
        self.total = 0
        self.drop = 0
        
    def forward(self, batch, training, key=None):
        self.total += 1
        if key is None:
            key = self.key
        if torch.rand(1) < self.p_uncond and training:
            self.drop += 1
            b = batch[key].shape[0]
            print(self.uncond_embedding.weight.data[0,:10])
            c = torch.tensor([0]*b)[:, None].cuda()
            c = self.uncond_embedding(c)
        else:
            # this is for use in crossattn
            c = batch[key][:, None]
            print(c.view(-1))
            c = self.embedding(c)
        print(self.drop/self.total)
        return c


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
                 multiplier=None,
                 size=None,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.size = size
        assert (size and not multiplier) or (multiplier and not size)
        if self.size:
            assert len(self.size) == self.n_stages
        elif self.multiplier:
            assert len(self.multiplier) == self.n_stages
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = False # out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            if self.multiplier:
                x = self.interpolator(x, scale_factor=self.multiplier[stage])
            else:
                x = self.interpolator(x, size=self.size[stage])

        if self.remap_output:
            x = self.channel_mapper(x)
        
        return x

    def encode(self, x):
        return self(x)


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

