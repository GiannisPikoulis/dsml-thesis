import cv2
import pickle
import os, sys
import numpy as np
import torchvision
import torch
import moviepy.editor as mpe
from PIL import Image
from einops import rearrange


def combine_audio_video(video_path, audio_path, output_path, fps=30):    
    video = mpe.VideoFileClip(video_path)
    video = video.set_audio(mpe.AudioFileClip(audio_path))
    video.write_videofile(output_path, fps=fps)


def main():
    
    os.makedirs('samples/videos', exist_ok=True)
    os.makedirs('samples/video_grids', exist_ok=True)
    os.makedirs('samples/motion_grids', exist_ok=True)
    os.makedirs('samples/identities', exist_ok=True)
    audio_template = '/gpu-data3/filby/MEAD/{subj}/audio/{emo}/{lvl}/{nbr}.wav'
    
    VIDEO_PKL = sys.argv[1]
    IDENTITY_PKL =  VIDEO_PKL.replace('video', 'identity')
#     MOTION_PKL = VIDEO_PKL.replace('video', 'motion')
    INFO_PKL = VIDEO_PKL.replace('video', 'info')
    CONFIG = VIDEO_PKL.split('cnfg=')[1].split('_')[0]
    CKPT = VIDEO_PKL.split('ckpt=')[1].split('.pkl')[0]
    ALIGN = VIDEO_PKL.split('align=')[1].split('_')[0]
    
    os.makedirs(f"samples/align={ALIGN}_config={CONFIG}_ckpt={CKPT}", exist_ok=True)
    os.makedirs(f"samples/align={ALIGN}_config={CONFIG}_ckpt={CKPT}/with_audio", exist_ok=True)

    with open(f'{VIDEO_PKL}', 'rb') as handle:
        video_pkl = pickle.load(handle)

    with open(f'{IDENTITY_PKL}', 'rb') as handle:
        identity_pkl = pickle.load(handle)

#     with open(f'{MOTION_PKL}', 'rb') as handle:
#         motion_pkl = pickle.load(handle)

    with open(f'{INFO_PKL}', 'rb') as handle:
        info_pkl = pickle.load(handle)

    for i in range(len(video_pkl)):
        subj, emo, lvl, nbr = info_pkl[i]        
        out_path = f"samples/align={ALIGN}_config={CONFIG}_ckpt={CKPT}/subj={subj}_emo={emo}_lvl={lvl}_nbr={nbr}.mp4"
        cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(out_path, cv2_fourcc, 30, (128, 128)) # output video name, fourcc, fps, size
        sample = video_pkl[i]
        sample = (255*sample).astype(np.uint8)
        sample = sample[:,:,:,::-1]

        for j in range(len(sample)): 
            video.write(sample[j]) 
        video.release()
        
        # Save combined audio with video
        out_path_with_audio = f"samples/align={ALIGN}_config={CONFIG}_ckpt={CKPT}/with_audio/subj={subj}_emo={emo}_lvl={lvl}_nbr={nbr}.mp4"
        audio_path = audio_template.format(subj=subj, emo=emo, lvl=lvl, nbr=nbr)
        combine_audio_video(video_path=out_path, audio_path=audio_path, output_path=out_path_with_audio, fps=30)
        
        # Save video frames as grid
        example = torch.tensor(video_pkl[i]).permute(0, 3, 1, 2)
        grid = torchvision.utils.make_grid(example, nrow=10)
        grid = 255. * rearrange(grid, 'c h w -> h w c').numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        img.save(f"samples/align={ALIGN}_config={CONFIG}_ckpt={CKPT}/grid_subj={subj}_emo={emo}_lvl={lvl}_nbr={nbr}.jpg")

#         # Save motion frames as grid
#         example = torch.tensor(motion_pkl[i]).permute(0, 3, 1, 2)
#         grid = torchvision.utils.make_grid(example, nrow=10)
#         grid = 255. * rearrange(grid, 'c h w -> h w c').numpy()
#         img = Image.fromarray(grid.astype(np.uint8))
#         img.save(f"samples/motion_grids/{ALIGN}_{PROP_ID}_{CONFIG}_{CKPT}_{subj}_{emo}_{lvl}_{nbr}.jpg")
        
        # Save motion frames as grid
        img = Image.fromarray((255*identity_pkl[i]).astype(np.uint8))
        img.save(f"samples/align={ALIGN}_config={CONFIG}_ckpt={CKPT}/identity_subj={subj}_emo={emo}_lvl={lvl}_nbr={nbr}.jpg")


if __name__ == '__main__':
    main()