from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import face_alignment
from abc import abstractmethod, ABC
import os, sys
import torch
import albumentations
import pickle
from PIL import Image


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
    with open('/gpu-data2/jpik/MEAD_v2/cropped_frames.txt', 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        
    face_detector = FAN()
    rescaler = albumentations.SmallestMaxSize(max_size=128)
    cropper = albumentations.CenterCrop(height=128, width=128)
    preprocessor = albumentations.Compose([rescaler, cropper])
    SKIP = set()
    cnt = 0
    for i in range(len(lines)):
        dir_path = '/'.join(lines[i].replace('video', 'landmarks').split('/')[:10])
        os.makedirs(dir_path, exist_ok=True)
        suffix = lines[i].split('/')[-1].split('.')[0] + '.pkl'
        save_path = os.path.join(dir_path, suffix)
        if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            cnt += 1
            continue
        print(f"Current row: {save_path} | Current index: {i} | cnt: {cnt}")
        img = Image.open(lines[i])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = preprocessor(image=img)["image"]
        try:
            _, _, landmarks = face_detector.run(img, with_landmarks=True)
            with open(save_path, 'wb') as handle:
                pickle.dump(landmarks[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
#             mouth = landmarks[0][48:68]
#             min_x, min_y = int(min(mouth[:, 0])), int(min(mouth[:, 1]))
#             img[min_y:,:,:] = 0
#             img = Image.fromarray(img)
#             img.save(save_path)
            cnt += 1
        except:
            SKIP.add(lines[i])
            continue
    
    with open('/gpu-data2/jpik/MEAD_v2/mask_skip.pkl', 'wb') as handle:
        pickle.dump(SKIP, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
if __name__ == "__main__":
    main()