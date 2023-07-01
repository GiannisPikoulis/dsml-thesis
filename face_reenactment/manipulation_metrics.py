import argparse
import os
import piq
import torch
import numpy as np
import lpips
import cv2
from skimage.io import imread
from hsemotion.facial_emotions import HSEmotionRecognizer
import pandas as pd


def psnr_2dirs(dir0, dir1):
    
    files = os.listdir(dir0)
    psnr_index = []
    cnt = 0

    for file in files:
#         print(f'Current file: {file}')
        if(os.path.exists(os.path.join(dir1, file))):
            cnt += 1
            # Load images
            x = torch.tensor(imread(os.path.join(dir0, file))).permute(2, 0, 1)[None, ...] / 255.
            y = torch.tensor(imread(os.path.join(dir1, file))).permute(2, 0, 1)[None, ...] / 255.

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            
            # Compute distance
            psnr_index.append(piq.psnr(x, y, data_range=1., reduction='none').item())
        
        else:
            raise ValueError("Given directories don't match")

#     print(f'Directory #1: {dir0}')
#     print(f'Directory #2: {dir1}')
#     print(f'Total image pairs processed: {cnt}')
#     print(f'Mean PSNR index: {np.mean(psnr_index)}')
    
    return np.mean(psnr_index)


def ssim_2dirs(dir0, dir1):
    
    files = os.listdir(dir0)
    ssim_index = []
    cnt = 0

    for file in files:
#         print(f'Current file: {file}')
        if(os.path.exists(os.path.join(dir1, file))):
            cnt += 1
            # Load images
            x = torch.tensor(imread(os.path.join(dir0, file))).permute(2, 0, 1)[None, ...] / 255.
            y = torch.tensor(imread(os.path.join(dir1, file))).permute(2, 0, 1)[None, ...] / 255.

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            
            # Compute distance
            ssim_index.append(piq.ssim(x, y, data_range=1.).item())
        
        else:
            raise ValueError("Given directories don't match")

#     print(f'Directory #1: {dir0}')
#     print(f'Directory #2: {dir1}')
#     print(f'Total image pairs processed: {cnt}')
#     print(f'Mean SSIM index: {np.mean(ssim_index)}')
    
    return np.mean(ssim_index)


def lpips_2dirs(dir0, dir1):
    
    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    if torch.cuda.is_available():
        loss_fn.cuda()
    
    files = os.listdir(dir0)
    cnt = 0
    dist = 0

    for file in files:
        if(os.path.exists(os.path.join(dir1, file))):
            cnt += 1
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(dir0, file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(dir1, file)))

            if torch.cuda.is_available():
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            dist += dist01.item()
#             print('%s: %.3f'%(file, dist01))
#             f.writelines('%s: %.6f\n'%(file, dist01))

#     print(f"Image pairs processed: {cnt}")
#     print(f"Mean LPIPS: {dist/cnt}")
    
    return dist/cnt


def run_emorec(dir0):
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    model_name = 'enet_b2_7'

    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    
    # directory
    samples_dir = str(dir0)
    y_trg = int(samples_dir.split('emotion=')[1].split('_')[0])  
    sample_paths = sorted([os.path.join(samples_dir, f) for f in os.listdir(samples_dir)])

    emotion2idx_dict = {"Neutral": 0, "Happiness": 1, "Sadness": 2, "Surprise": 3, "Fear": 4, "Disgust": 5, "Anger": 6}
    idx2emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger"}

    counts = dict.fromkeys(emotion2idx_dict.keys(), 0)
    for i, path in enumerate(sample_paths):
#         print(f'Index: {i}, Path: {path}')
        frame_bgr = cv2.imread(path)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(frame, logits=True)
        counts[emotion] += 1

#     print(f'Target emotion: {y_trg}')
#     print(f'Predictions: {counts}')
    
    for k in counts.keys():
        counts[k] = np.round(counts[k] / len(sample_paths), 3)
        
    return counts[idx2emotion_dict[y_trg]], counts

if __name__ == '__main__':
    manipulated_emotion_dir_paths = sorted([os.path.join('samples/manipulations', DIR)  for DIR in os.listdir('samples/manipulations')])

    steps_list = []
    scale_list = []
    strength_list = []
    emotion_list = []
    lpips_list = []
    psnr_list = []
    ssim_list = []
    predictions_list = []
    target_percentage_list = []

    for emodir in manipulated_emotion_dir_paths:
        
        dir_xrecs = sorted([os.path.join(emodir, f) for f in os.listdir(emodir) if os.path.isdir(os.path.join(emodir, f)) and 'xrec' in f])
        dir_x0 = sorted([os.path.join(emodir, f) for f in os.listdir(emodir) if os.path.isdir(os.path.join(emodir, f)) and 'x0' in f])[0]
        
        for dir_xrec in dir_xrecs:
            lpips_metric = lpips_2dirs(dir_x0, dir_xrec)
            ssim_metric = ssim_2dirs(dir_x0, dir_xrec)
            psnr_metric = psnr_2dirs(dir_x0, dir_xrec)
            target_percentage, predicted = run_emorec(dir_xrec)
            
            steps = dir_xrec.split('_')[3]
            steps = int(steps.split('ddim')[1])
            scale, strength, emotion = dir_xrec.split('_')[5:8]
            scale = float(scale.split('scale')[1])
            strength = float(strength.split('strength')[1])
            emotion = int(emotion.split('emotion=')[1])
            
            steps_list.append(steps)
            scale_list.append(scale)
            strength_list.append(strength)
            emotion_list.append(emotion)
            lpips_list.append(lpips_metric)
            psnr_list.append(psnr_metric)
            ssim_list.append(ssim_metric)
            predictions_list.append(predicted)
            target_percentage_list.append(target_percentage)
            
            print('DDIM Steps: {}, Scale: {}, Strength: {}, Label: {}, LPIPS: {}, SSIM: {}, PSNR: {}, Target: {}, All: {}'.format(steps, scale, strength, emotion, lpips_metric, ssim_metric, psnr_metric, target_percentage, predicted))

    df = pd.DataFrame({'steps': pd.Series(steps_list), 'scale': pd.Series(scale_list), 'strength': pd.Series(strength_list), 
                    'emotion': pd.Series(emotion_list), 'lpips': pd.Series(lpips_list), 'ssim': pd.Series(ssim_list), 
                    'psnr': pd.Series(psnr_list), 'emorec_percentage': pd.Series(target_percentage_list)})

    df = pd.concat([df, pd.DataFrame(predictions_list)], axis=1)
    df.to_csv('emotion_manipulation_ablation_results.csv')