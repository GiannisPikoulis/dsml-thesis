import numpy as np
import pickle
import os
import sys
import cv2
from skimage.io import imread


def load_pickle(path):
    if os.path.getsize(path) > 0:
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
    else:
        return None

    
def resize_images(images, new_size):
    # Get the desired height (new_size[0]) and width (new_size[1])
    new_height, new_width = new_size

    # Initialize an empty list to store the resized images
    resized_images = []

    # Iterate over each image in the input array
    for image in images:
        # Resize the image using cv2.resize()
        resized_image = cv2.resize(image, (new_width, new_height))

        # Append the resized image to the list
        resized_images.append(resized_image)

    # Convert the list of resized images to a numpy array
    resized_images = np.array(resized_images)

    return resized_images    


def cut_mouth(images, landmarks, _window_margin=12, _start_idx=48, _stop_idx=68, _crop_height=72, _crop_width=72):
# function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages

    images = images.transpose((0,3,1,2))
    mouth_sequence = []

    for frame_idx, frame in enumerate(images):
        window_margin = min(_window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = landmarks[frame_idx-_window_margin:frame_idx+window_margin+1].mean(axis=0)
        smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
        center_x, center_y = np.mean(landmarks[frame_idx][_start_idx:_stop_idx], axis=0)

        center_x = center_x.round()
        center_y = center_y.round()

        height = _crop_height // 2
        width = _crop_width // 2
        
        img = frame

        threshold = 5

        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        if center_y + height > img.shape[-2]:
            center_y = img.shape[-2] - height
        if center_y + height > img.shape[-2] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[-1]:
            center_x = img.shape[-1] - width
        if center_x + width > img.shape[-1] + threshold:
            raise Exception('too much bias in width')
        
        mouth = img[...,int(center_y - height): int(center_y + height),
                        int(center_x - width): int(center_x + round(width))]

        mouth_sequence.append(mouth)

    mouth_sequence = np.stack(mouth_sequence, axis=0)
    return mouth_sequence.transpose((0,2,3,1))



if __name__ == "__main__":
    
    VIDEO_PKL = sys.argv[1]
    INFO_PKL = VIDEO_PKL.replace('video', 'info')
    CONFIG = VIDEO_PKL.split('cnfg=')[1].split('_')[0]
    CKPT = VIDEO_PKL.split('ckpt=')[1].split('.pkl')[0]
    ALIGN = VIDEO_PKL.split('align=')[1].split('_')[0]
    
    generated_frames = load_pickle(VIDEO_PKL)
    info = load_pickle(INFO_PKL)
    RESOLUTION = 88
    FPS = 30
    
    landmarks_template = "/gpu-data2/jpik/MEAD_v2/{subj}/landmarks/front/{emotion}/{lvl}/{nbr}"
    clip_template = "/gpu-data2/jpik/MEAD_v2/{subj}/video/front/{emotion}/{lvl}/{nbr}"
    
    for idx, (subj, emotion, lvl, nbr) in enumerate(info):
        landmark_path = landmarks_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        clip_path = clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        frames = sorted(os.listdir(clip_path))
        assert len(frames) == generated_frames[idx].shape[0]
        
        all_frames = list()
        l_list = list()
        for i, frame in enumerate(frames):
            img = imread(os.path.join(clip_path, frame))/255.
            all_frames.append(img)
            l = load_pickle(os.path.join(landmark_path, frame.replace('jpg', 'pkl')))
            l_list.append(l)
        all_frames = np.stack(all_frames, axis=0)
        all_frames = resize_images(all_frames, (128, 128)) # resize to 128x128
        l_list = np.stack(l_list, axis=0)

        a = cut_mouth(all_frames, l_list)
        a = resize_images(a, (RESOLUTION, RESOLUTION))
        
        # Save each mouth crops as videos
        os.makedirs(f'samples/mouth_crops_gt/align={ALIGN}_config={CONFIG}_ckpt={CKPT}', exist_ok=True)
        out_path = f"samples/mouth_crops_gt/align={ALIGN}_config={CONFIG}_ckpt={CKPT}/subj={subj}_emo={emotion}_lvl={lvl}_nbr={nbr}.mp4"
        cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(out_path, cv2_fourcc, FPS, (RESOLUTION, RESOLUTION)) # output video name, fourcc, fps, size
        sample = (255*a).astype(np.uint8)
        sample = sample[:,:,:,::-1]

        for j in range(len(sample)): 
            video.write(sample[j]) 
        video.release()