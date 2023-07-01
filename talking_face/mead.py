import numpy as np
import scipy.io.wavfile as wav
import librosa
import os, sys, shutil, argparse, copy, pickle
import math, scipy
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

WAV2VEC_CONFIG = "facebook/wav2vec2-base-960h"

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
        
    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
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
            print('------------- WARNING 2 -------------')
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
    
    
def main():
    
    os.makedirs('/gpu-data2/jpik/MEAD/precomputed_audio_features', exist_ok=True)
    with open('/gpu-data2/jpik/MEAD/cropped_frames.txt', 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    FRAME_TUPLES = set([(l.split('/')[4], l.split('/')[7], l.split('/')[8], l.split('/')[9]) for l in lines])

    with open('/gpu-data2/jpik/MEAD/mead_skip.pkl', 'rb') as handle:
        SKIPS = pickle.load(handle)
    TO_SKIP = set(list(set([(l.split('/')[4], l.split('/')[7], l.split('/')[8], l.split('/')[9]) for l in SKIPS])))

    AUDIO_PATH = '/gpu-data3/filby/MEAD'
    with open('/gpu-data2/jpik/MEAD/audio.pkl', 'rb') as handle:
        AUDIO_TUPLES = set((list(pickle.load(handle))))
    AUDIO_TUPLES = sorted(list((AUDIO_TUPLES & FRAME_TUPLES) - TO_SKIP))
    
    # wav2vec 2.0 weights initialization
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_CONFIG)
    audio_encoder = Wav2Vec2Model.from_pretrained(WAV2VEC_CONFIG)
    audio_encoder._freeze_all()
    audio_encoder.to('cuda')
    audio_encoder.eval()
    
    assert sum(p.numel() for p in audio_encoder.parameters() if p.requires_grad) == 0
    cnt = 0
#     feature_dict = {}
    for i in range(len(AUDIO_TUPLES)):
        subject, emotion, lvl, clip = AUDIO_TUPLES[i]
#         if (subject, emotion, lvl, clip) not in feature_dict:
#             feature_dict[(subject, emotion, lvl, clip)] = None
        wav_path = os.path.join(AUDIO_PATH, subject, 'audio', emotion, lvl, f'{clip}.wav')
        print(f"Processed audio file: {wav_path}")
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature,(-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')
        num_frames = len(os.listdir(f"/gpu-data2/jpik/MEAD/{subject}/video/front/{emotion}/{lvl}/{clip}"))
        x = audio_encoder(audio_feature, frame_num=num_frames).last_hidden_state.squeeze(0).cpu().detach().numpy()
        assert x.shape[0] == num_frames
        
#         feature_dict[(subject, emotion, lvl, clip)] = x
        with open(f'/gpu-data2/jpik/MEAD/precomputed_audio_features/{subject}_{emotion}_{lvl}_{clip}.pkl', 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cnt += 1
        print(f"Total processed: {cnt}")
        
#     with open('/gpu-data2/jpik/MEAD/wav2vec_audio_features.pkl', 'wb') as handle:
#         pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()