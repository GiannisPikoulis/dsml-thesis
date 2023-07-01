import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os, sys, glob
from PIL import Image
from taming.data.custom import AffectnetTrain, AffectnetTest
from abc import abstractmethod, ABC
import face_alignment
from skimage.io import imread, imsave
from skimage.transform import rescale, estimate_transform, warp, resize

SKIP = set()

class FaceDetector(ABC):

    @abstractmethod
    def run(self, image, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)


class FAN(FaceDetector):

    def __init__(self, device = 'cuda', threshold=0.8):
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                  device=str(device),
                                                  flip_input=False,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)

    # @profile
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out # attempt to prevent memory leaks
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'


def main():
    
    SAVE_DIR = '/gpu-data2/jpik/MEAD_v2'
    os.makedirs(SAVE_DIR, exist_ok=True)
    IMAGE_SIZE = 224
    
    filbyFiles = glob.glob('/gpu-data3/filby/MEAD/*/video/front/*/*/'+'[0-9]'*3)
    filbyDirs = [f for f in filbyFiles if os.path.isdir(f)]
        
    face_detector = FAN()
    cnt = 0
    for j, src_dir in enumerate(sorted(filbyDirs)):

#         if j > 0:
#             break
            
        subj, type_, angle, emotion, lvl, clip = src_dir.split('/')[4:10]
        res_dir = os.path.join(SAVE_DIR, subj, type_, angle, emotion, lvl, clip)
        print(f'SRC_DIR: {src_dir} | RES_DIR: {res_dir}')
        os.makedirs(res_dir, exist_ok=True)
        
        done = False
        x1 = None
        x2 = None
        y1 = None
        y2 = None
        src_frames = sorted(os.listdir(src_dir))
        for i, src_frame in enumerate(src_frames):
            src_path = os.path.join(src_dir, src_frame)
            res_path = os.path.join(res_dir, src_frame)
            
            if not done:
                try:
                    img = np.array(imread(src_path))
                    resized_img = (resize(img, (216, 384))*255).astype(np.uint8)    
                    bbox, bbox_type, _ = face_detector.run(resized_img, with_landmarks=True)
                    if len(bbox) > 0:
                        x1, y1, x2, y2 = [max(int(d),0) for d in bbox[0]]
                        x1 = x1-10
                        y1 = y1-10
                        x2 = x2+10
                        y2 = y2+10
                        offset = (max(x2-x1, y2-y1) - min(x2-x1, y2-y1)) // 2
                        if x2-x1 < y2-y1:
                            x1 = max(x1-offset,0)
                            x2 = x2 + offset
                        elif x2-x1 > y2-y1:
                            y1 = max(y1-offset,0)
                            y2 = y2 + offset
                        else:
                            pass
                        done = True
                        print(f'BBOX calculated based on frame #{i+1}/{len(src_frames)}')
                    else:
                        continue
                except:
                    SKIP.add(src_path)
                    continue
            else:
                break
        
        for i, src_frame in enumerate(src_frames):
            src_path = os.path.join(src_dir, src_frame)
            res_path = os.path.join(res_dir, src_frame)
                
            try:
                img = np.array(imread(src_path))
                resized_img = (resize(img, (216, 384))*255).astype(np.uint8)
                resized_img = resized_img[y1:y2,x1:x2]
            except:
                SKIP.add(src_path)
                continue
            
            if os.path.isfile(res_path):
                cnt += 1
                continue
                
            res = (resize(resized_img, (IMAGE_SIZE, IMAGE_SIZE))*255).astype(np.uint8)  
            imsave(res_path, res)
            cnt += 1
            print(f'Frames cropped: {cnt}')
            
    with open('/gpu-data2/jpik/MEAD_v2/mask_skip.pkl', 'wb') as handle:
        pickle.dump(SKIP, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
if __name__ == '__main__':
    main()