import os, sys
import piq
import torch
import numpy as np
import lpips
from skimage.io import imread
import pickle
import glob
import torchvision.transforms.functional as F_v


def load_pickle(path):
    if os.path.getsize(path) > 0:
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
    else:
        return None  
    
    
def calculate_psnr(gt_frames, generated_frames):
    
    psnr_index = []

    for i, frame_path in enumerate(gt_frames):
        # Load images
        x = torch.tensor(imread(frame_path)).permute(2, 0, 1)[None, ...] / 255.
        x = F_v.resize(x, 128)
        y = torch.tensor(generated_frames[i]).permute(2, 0, 1)[None, ...] # already in range [0, 1]

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            
        # Compute distance
        psnr_index.append(piq.psnr(x, y, data_range=1., reduction='none').item())
    
    return np.mean(psnr_index)


def calculate_ssim(gt_frames, generated_frames):
    
    ssim_index = []
    
    for i, frame_path in enumerate(gt_frames):
        # Load images
        x = torch.tensor(imread(frame_path)).permute(2, 0, 1)[None, ...] / 255.
        x = F_v.resize(x, 128)
        y = torch.tensor(generated_frames[i]).permute(2, 0, 1)[None, ...] # already in range [0, 1]
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            
        # Compute distance
        ssim_index.append(piq.ssim(x, y, data_range=1.).item()) 
        
    return np.mean(ssim_index)


def calculate_lpips(gt_frames, generated_frames):
    
    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    if torch.cuda.is_available():
        loss_fn.cuda()
    
    dist = 0
    
    for i, frame_path in enumerate(gt_frames):
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(frame_path)) # RGB image from [-1,1]
        img0 = F_v.resize(img0, 128)
        img1 = lpips.im2tensor(generated_frames[i], factor=0.5)

        if torch.cuda.is_available():
            img0 = img0.cuda()
            img1 = img1.cuda()

        # Compute distance
        dist01 = loss_fn.forward(img0, img1)
        dist += dist01.item()
    
    return dist / len(gt_frames)  


if __name__ == '__main__':
    
    generated_video_pkl = str(sys.argv[1]) # .pkl
    generated_info_pkl = generated_video_pkl.replace('video', 'info') 
    generated_videos = load_pickle(generated_video_pkl)
    info_tuples = load_pickle(generated_info_pkl)
    clip_template = "/gpu-data2/jpik/MEAD_v2/{subj}/video/front/{emotion}/{lvl}/{nbr}"
    
    psnr_ = 0 
    ssim_ = 0 
    lpips_ = 0
    for i, (subj, emotion, lvl, nbr) in enumerate(info_tuples):
        gt_path = clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        gt_frames = sorted(glob.glob(f'{gt_path}/*.jpg'))
        assert len(gt_frames) == generated_videos[i].shape[0]
        psnr_ += calculate_psnr(gt_frames, generated_videos[i])
        ssim_ += calculate_ssim(gt_frames, generated_videos[i])
        lpips_ += calculate_lpips(gt_frames, generated_videos[i])
        
    psnr_, ssim_, lpips_ = psnr_/len(info_tuples), ssim_/len(info_tuples), lpips_/len(info_tuples)
    
    with open('/gpu-data2/jpik/difftalk/samples/metric_logs.txt', 'a') as f:
        f.write(f'{generated_video_pkl}: PSNR={psnr_}, SSIM={ssim_}, LPIPS={lpips_}\n') 