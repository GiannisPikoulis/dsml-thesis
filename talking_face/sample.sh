#!/bin/bash

echo "Running on GPU devices with IDs: $1"

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-05-28T11-25-06_mead-128-ldm-f4-v7_audio_emotion_w4_bs32/checkpoints/epoch=000128-train_loss=0.018-val_loss_ema=0.024.ckpt True 200 1.0 1.0 150 audio_emotion_w4_bs32 4

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-05-28T11-25-06_mead-128-ldm-f4-v7_audio_emotion_w4_bs32/checkpoints/epoch=000128-train_loss=0.018-val_loss_ema=0.024.ckpt True 200 0.0 1.0 5 audio_emotion_w4_bs32 4

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-05-30T16-29-22_mead-128-ldm-f4-v7_audio_emotion_w8_bs32/checkpoints/epoch=000114-train_loss=0.019-val_loss_ema=0.024.ckpt True 200 1.0 1.0 5 audio_emotion_w8_bs32 8

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-05-30T16-29-22_mead-128-ldm-f4-v7_audio_emotion_w8_bs32/checkpoints/epoch=000114-train_loss=0.019-val_loss_ema=0.024.ckpt True 200 0.0 1.0 5 audio_emotion_w8_bs32 8

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-05-28T13-45-23_mead-128-ldm-f4-v7_audio_emotion_w2_bs32/checkpoints/epoch=000128-train_loss=0.018-val_loss_ema=0.024.ckpt True 200 1.0 1.0 5 audio_emotion_w2_bs32 2

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-05-28T13-45-23_mead-128-ldm-f4-v7_audio_emotion_w2_bs32/checkpoints/epoch=000128-train_loss=0.018-val_loss_ema=0.024.ckpt True 200 0.0 1.0 5 audio_emotion_w2_bs32 2

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-06-01T09-16-53_mead-128-ldm-f4-v7_audio_emotion_w1_bs32/checkpoints/epoch=000114-train_loss=0.019-val_loss_ema=0.024.ckpt True 200 1.0 1.0 5 audio_emotion_w1_bs32 1

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v7.yaml /gpu-data2/jpik/difftalk/logs/2023-06-01T09-16-53_mead-128-ldm-f4-v7_audio_emotion_w1_bs32/checkpoints/epoch=000114-train_loss=0.019-val_loss_ema=0.024.ckpt True 200 0.0 1.0 5 audio_emotion_w1_bs32 1

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v11.yaml  /gpu-data2/jpik/difftalk/logs/2023-06-01T09-33-06_mead-128-ldm-f4-v11_audio_emotion_w4_b8_ft/checkpoints/epoch=000035-train_loss=0.090-val_loss_ema=0.211.ckpt True 200 1.0 1.0 150 audio_emotion_w4_bs8_ft 4

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v13.yaml  /gpu-data2/jpik/difftalk/logs/2023-06-03T01-21-04_mead-128-ldm-f4-v13_audio_emotion_w4_bs8_ft_range/checkpoints/epoch=000045-train_loss=0.029-val_loss_ema=0.068.ckpt True 200 1.0 1.0 5 audio_emotion_w4_bs8_ft_range 4

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk_audio_only.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v17.yaml /gpu-data2/jpik/difftalk/logs/2023-06-09T00-45-16_mead-128-ldm-f4-v17_audio_bundle_w8_bs32/checkpoints/epoch=000128-train_loss=0.018-val_loss_ema=0.024.ckpt True 200 1.0 1.0 150 audio_bundle_w8_bs32 8

CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk_identity_only.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v18.yaml /gpu-data2/jpik/difftalk/logs/2023-06-10T18-39-06_mead-128-ldm-f4-v18/checkpoints/epoch=000128-train_loss=0.030-val_loss_ema=0.035.ckpt True 200 1.0 1.0 150 audio_emotion_identity_only_w4_bs32 4

############################################################################################################################################

# Identity only, precomputed audio
# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v2.yaml /gpu-data2/jpik/difftalk/logs/2023-04-14T20-21-35_mead-128-ldm-f4-v2_cont1/checkpoints/epoch=000183-train_loss=0.031-val_loss_ema=0.039.ckpt True 200 1.0 1.0 5

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v2.yaml /gpu-data2/jpik/difftalk/logs/2023-04-14T20-21-35_mead-128-ldm-f4-v2_cont1/checkpoints/epoch=000183-train_loss=0.031-val_loss_ema=0.039.ckpt True 200 0.0 1.0 5


# Identity only, trainable audio
# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_trainable.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v3.yaml /gpu-data2/jpik/difftalk/logs/2023-04-22T03-15-47_mead-128-ldm-f4-v3_cont1/checkpoints/epoch=000054-train_loss=0.041-val_loss_ema=0.044.ckpt False 200 1.0 1.0 5

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_trainable.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v3.yaml /gpu-data2/jpik/difftalk/logs/2023-04-22T03-15-47_mead-128-ldm-f4-v3_cont1/checkpoints/epoch=000054-train_loss=0.041-val_loss_ema=0.044.ckpt False 200 0.0 1.0 5

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_trainable.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v3.yaml /gpu-data2/jpik/difftalk/logs/2023-04-22T03-15-47_mead-128-ldm-f4-v3_cont1/checkpoints/epoch=000054-train_loss=0.041-val_loss_ema=0.044.ckpt False 200 0.0 1.0 5


# DiffTalk
# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v4.yaml /gpu-data2/jpik/difftalk/logs/2023-04-27T21-54-34_mead-128-ldm-f4-v4/checkpoints/epoch=000086-train_loss=0.020-val_loss_ema=0.024.ckpt False 200 0.0 1.0 5 difftalk

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v4.yaml /gpu-data2/jpik/difftalk/logs/2023-04-27T21-54-34_mead-128-ldm-f4-v4/checkpoints/epoch=000086-train_loss=0.020-val_loss_ema=0.024.ckpt False 200 1.0 1.0 5 difftalk

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v4.yaml /gpu-data2/jpik/difftalk/logs/2023-05-01T16-59-41_mead-128-ldm-f4-v4_cont1/checkpoints/epoch=000134-train_loss=0.018-val_loss_ema=0.024.ckpt False 200 0.0 1.0 5 difftalk

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_difftalk.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v4.yaml /gpu-data2/jpik/difftalk/logs/2023-05-01T16-59-41_mead-128-ldm-f4-v4_cont1/checkpoints/epoch=000134-train_loss=0.018-val_loss_ema=0.024.ckpt False 200 1.0 1.0 5 difftalk


# identity + identity loss
# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_frozen.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v6.yaml /gpu-data2/jpik/difftalk/logs/2023-05-11T14-59-36_mead-128-ldm-f4-v6_cont1/checkpoints/epoch=000060-train_loss=0.068-val_loss_ema=0.063.ckpt False 200 0.0 1.0 1 idloss True False

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_frozen.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v6.yaml /gpu-data2/jpik/difftalk/logs/2023-05-11T14-59-36_mead-128-ldm-f4-v6_cont1/checkpoints/epoch=000060-train_loss=0.068-val_loss_ema=0.063.ckpt False 200 0.0 1.0 1 idloss True True

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_frozen.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v6.yaml /gpu-data2/jpik/difftalk/logs/2023-05-11T14-59-36_mead-128-ldm-f4-v6_cont1/checkpoints/epoch=000060-train_loss=0.068-val_loss_ema=0.063.ckpt False 200 0.0 1.0 1 idloss False False

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_frozen.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v6.yaml /gpu-data2/jpik/difftalk/logs/2023-05-11T14-59-36_mead-128-ldm-f4-v6_cont1/checkpoints/epoch=000060-train_loss=0.068-val_loss_ema=0.063.ckpt False 200 0.0 1.0 1 idloss False True

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_frozen.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v6.yaml /gpu-data2/jpik/difftalk/logs/2023-05-11T14-59-36_mead-128-ldm-f4-v6_cont1/checkpoints/epoch=000060-train_loss=0.068-val_loss_ema=0.063.ckpt False 200 1.0 1.0 1 idloss True False

# CUDA_VISIBLE_DEVICES=$1 python progressive_sampling_no_motion_frozen.py /gpu-data2/jpik/difftalk/configs/latent-diffusion/mead-128-ldm-f4-v6.yaml /gpu-data2/jpik/difftalk/logs/2023-05-11T14-59-36_mead-128-ldm-f4-v6_cont1/checkpoints/epoch=000060-train_loss=0.068-val_loss_ema=0.063.ckpt False 200 1.0 1.0 1 idloss True True