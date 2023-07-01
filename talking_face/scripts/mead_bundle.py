import numpy as np
import scipy.io.wavfile as wav
import librosa
import os, sys, shutil, argparse, copy, pickle
import math, scipy
from typing import Optional, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

    
def main():
    
    os.makedirs('/gpu-data2/jpik/MEAD/bundle_audio_features', exist_ok=True)
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
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    model = bundle.get_model().to(device)    
    
    cnt = 0
    for i in range(len(AUDIO_TUPLES)):
        subject, emotion, lvl, clip = AUDIO_TUPLES[i]
        wav_path = os.path.join(AUDIO_PATH, subject, 'audio', emotion, lvl, f'{clip}.wav')
        print(f"Processed audio file: {wav_path}")
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.to(device)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        with torch.inference_mode():
            emission, _ = model(waveform)
        num_frames = len(os.listdir(f"/gpu-data2/jpik/MEAD/{subject}/video/front/{emotion}/{lvl}/{clip}"))
        x = linear_interpolation(emission[0].unsqueeze(0), 49, 30, output_len=num_frames).squeeze(0).cpu().numpy()
        assert x.shape[0] == num_frames
        
        with open(f'/gpu-data2/jpik/MEAD/bundle_audio_features/{subject}_{emotion}_{lvl}_{clip}.pkl', 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cnt += 1
        print(f"Total processed: {cnt}")


if __name__ == "__main__":
    main()