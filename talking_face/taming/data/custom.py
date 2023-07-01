import os
import numpy as np
import albumentations
import pickle
from torch.utils.data import Dataset
import random
from PIL import Image
import torch
from einops import rearrange
from transformers import Wav2Vec2Processor
import librosa

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex, AffectnetPaths

WAV2VEC_CONFIG = "facebook/wav2vec2-base-960h"


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

    
class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        
        
class AffectnetTrain(CustomBase):
    def __init__(self, size, training_images_list_file, model=None, mode=None):
        assert model in [None, 'deca', 'emoca']
        assert mode in [None, 'train', 'val']
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = AffectnetPaths(paths=paths, size=size, random_crop=False, model=model, mode=mode)


class AffectnetTest(CustomBase):
    def __init__(self, size, test_images_list_file, model=None, mode=None):
        assert model in [None, 'deca', 'emoca']
        assert mode in [None, 'train', 'val']
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = AffectnetPaths(paths=paths, size=size, random_crop=False, model=model, mode=mode)
        
# Identity only, frozen WAV2VEC2
class MEADBase(Dataset):
    def __init__(self, size=128, tuples_path=None, 
                 random_crop=False, mode='train', 
                 max_shortcut=30, force_align=False,
                 audio_window=4):
        
        assert size is not None and tuples_path is not None
        assert mode in ['train', 'sample']
        self.size = size
        self.max_shortcut = max_shortcut
        self.random_crop = random_crop
        self.force_align = force_align
        self.audio_window = audio_window
        self.mode = mode
        self.clip_template = "/gpu-data2/jpik/MEAD_v2/{subj}/video/front/{emotion}/{lvl}/{nbr}"
#         self.gray_template = "/gpu-data2/jpik/MEAD/{subj}/grayscale/front/{emotion}/{lvl}/{nbr}"
        self.audio_dir = "/gpu-data2/jpik/MEAD/precomputed_audio_features"
        self.emotion2label = {"angry": 6, "contempt": 7, "disgusted": 5, "fear": 4, "happy": 1, "neutral": 0, "sad": 2, "surprised": 3}
        self.label2emotion = {v: k for k, v in self.emotion2label.items()}
        
        with open(tuples_path, 'rb') as handle:
            self.tuples = sorted(list(pickle.load(handle)))
            
#         self.noiser = albumentations.augmentations.transforms.GaussNoise(var_limit=1)
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def fill_frames(self, current_idx, substitute_idx):
        if current_idx < 0:
            return substitute_idx
        else:
            return current_idx
    
    def __getitem__(self, idx):
        example = dict()
        subj, emotion, lvl, nbr = self.tuples[idx]
        clip_path = self.clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        audio_path = os.path.join(self.audio_dir, f"{subj}_{emotion}_{lvl}_{nbr}.pkl")
        audio_features = self.load_pickle(audio_path)
        frames = sorted(os.listdir(clip_path))
        assert len(frames) == audio_features.shape[0]
        frame_indices = range(len(frames))
        
        # Target frame
        if self.mode == 'train':
            anchor_idx = random.choice(frame_indices) # during training, the target frame is chosen randomly
        elif self.mode == 'sample':
            anchor_idx = 0 # during sampling, the initial target frame matches the first frame of the sequence        
        image = self.preprocess_image(os.path.join(clip_path, frames[anchor_idx]))
        
        # Identity frame: [0,...,min(t+30,T)]
        id_idx = random.choice(range(min(len(frames), anchor_idx+self.max_shortcut)))
        if self.mode == 'sample' and self.force_align:
            print('------------ FORCED ALIGNMENT ------------')
            id_idx = 0
        id_frame = self.preprocess_image(os.path.join(clip_path, frames[id_idx]))
        
#         # Motion frames: t-2, t-1
#         motion = np.array([self.preprocess_image(os.path.join(gray_path, frames[self.fill_frames(anchor_idx+i, id_idx)])) for i in range(-2, 0)])

        # Audio
        if self.mode == 'train':
            assert [min(max(anchor_idx+i, 0), len(audio_features)-1) for i in range(-self.audio_window, self.audio_window+1)][self.audio_window] == anchor_idx
            audio = np.array([audio_features[min(max(anchor_idx+i, 0), len(audio_features)-1)] for i in range(-self.audio_window, self.audio_window+1)])
        elif self.mode == 'sample':
            audio = np.array(audio_features)
            assert len(frames) == len(audio_features)
        
        # Construct output dict
        example["image"] = image
        example["identity"] = id_frame
        example["audio"] = audio
#         if self.mode == 'train':
#             example["motion"] = motion
#         if self.mode == 'sample':
#             example["audio_features"] = audio_features
#         elif self.mode == 'train':
#             example["audio"] = audio
        example["class_label"] = self.emotion2label[emotion]
        example["human_label"] = emotion
        example["anchor_path"] = os.path.join(clip_path, frames[anchor_idx])
        example["frame_idx"] = anchor_idx
        example["identity_idx"] = id_idx
        example["num_frames"] = len(frames)
        example["subj"] = subj
        example["lvl"] = lvl
        example["nbr"] = nbr
        return example
           
    def __len__(self):
        return len(self.tuples)
    
# Identity only, trainbable WAV2VEC2
class MEADBase2(Dataset):
    def __init__(self, size=128, tuples_path=None, 
                 random_crop=False, mode='train', 
                 max_shortcut=30, force_align=False):
        
        assert size is not None and tuples_path is not None
        assert mode in ['train', 'sample']
        self.size = size
        self.max_shortcut = max_shortcut
        self.random_crop = random_crop
        self.force_align = force_align
        self.mode = mode
        self.clip_template = "/gpu-data2/jpik/MEAD/{subj}/video/front/{emotion}/{lvl}/{nbr}"
        self.audio_template = "/gpu-data3/filby/MEAD/{subj}/audio/{emotion}/{lvl}/{nbr}.wav"
        self.audio_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_CONFIG)
        self.emotion2label = {"angry": 6, "contempt": 7, "disgusted": 5, "fear": 4, "happy": 1, "neutral": 0, "sad": 2, "surprised": 3}
        self.label2emotion = {v: k for k, v in self.emotion2label.items()}
        
        with open(tuples_path, 'rb') as handle:
            self.tuples = sorted(list(pickle.load(handle)))
            
        self.noiser = albumentations.augmentations.transforms.GaussNoise(var_limit=1)
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def fill_frames(self, current_idx, substitute_idx):
        if current_idx < 0:
            return substitute_idx
        else:
            return current_idx
    
    def __getitem__(self, idx):
        example = dict()
        subj, emotion, lvl, nbr = self.tuples[idx]
        wav_path = self.audio_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        clip_path = self.clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        frames = sorted(os.listdir(clip_path))
        frame_indices = range(len(frames))
        
        # target frame
        if self.mode == 'train':
            anchor_idx = random.choice(frame_indices) # during training, the target frame is chosen randomly
        elif self.mode == 'sample':
            anchor_idx = 0 # during sampling, the initial target frame matches the first frame of the sequence
        image = self.preprocess_image(os.path.join(clip_path, frames[anchor_idx]))
        
        # Identity frame: [0,...,min(t+30,T)]
        id_idx = random.choice(range(min(len(frames), anchor_idx+self.max_shortcut)))
        if self.mode == 'sample' and self.force_align:
            print('------------ FORCED ALIGNMENT ------------')
            id_idx = 0
        id_frame = self.preprocess_image(os.path.join(clip_path, frames[id_idx]))

        # Audio
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        audio_feature = np.squeeze(self.audio_processor(speech_array, sampling_rate=16000).input_values)
        
        # Construct output dict
        example["image"] = image
        example["identity"] = id_frame
        example["audio"] = audio_feature
        example["class_label"] = self.emotion2label[emotion]
        example["human_label"] = emotion
        example["anchor_path"] = os.path.join(clip_path, frames[anchor_idx])
        example["frame_idx"] = anchor_idx
        example["identity_idx"] = id_idx
        example["num_frames"] = len(frames)
        example["subj"] = subj
        example["lvl"] = lvl
        example["nbr"] = nbr
        return example
           
    def __len__(self):
        return len(self.tuples)
    
# DiffTalk, frozen WAV2VEC2    
class MEADBase3(Dataset):
    def __init__(self, audio_window, size=128, tuples_path=None, 
                 random_crop=False, mode='train', 
                 max_shortcut=60, force_align=False):
        
        assert size is not None and tuples_path is not None
        assert mode in ['train', 'sample']
        self.size = size
        self.max_shortcut = max_shortcut
        self.random_crop = random_crop
        self.force_align = force_align
        self.audio_window = audio_window
        self.mode = mode
        self.clip_template = "/gpu-data2/jpik/MEAD_v2/{subj}/video/front/{emotion}/{lvl}/{nbr}"
        self.landmarks_template = "/gpu-data2/jpik/MEAD_v2/{subj}/landmarks/front/{emotion}/{lvl}/{nbr}"
        self.audio_dir = "/gpu-data2/jpik/MEAD/precomputed_audio_features"
        self.emotion2label = {"angry": 6, "contempt": 7, "disgusted": 5, "fear": 4, "happy": 1, "neutral": 0, "sad": 2, "surprised": 3}
        self.label2emotion = {v: k for k, v in self.emotion2label.items()}
        
        with open(tuples_path, 'rb') as handle:
            self.tuples = sorted(list(pickle.load(handle)))
            
        self.noiser = albumentations.augmentations.transforms.GaussNoise(var_limit=1)
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def load_pickle(self, path):
        if os.path.getsize(path) > 0:
            with open(path, 'rb') as handle:
                b = pickle.load(handle)
            return b
        else:
            return None
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def fill_frames(self, current_idx, substitute_idx):
        if current_idx < 0:
            return substitute_idx
        else:
            return current_idx
    
    def __getitem__(self, idx):
        example = dict()
        subj, emotion, lvl, nbr = self.tuples[idx]
        clip_path = self.clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        landmarks_path = self.landmarks_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        audio_path = os.path.join(self.audio_dir, f"{subj}_{emotion}_{lvl}_{nbr}.pkl")
        audio_features = self.load_pickle(audio_path)
        frames = sorted(os.listdir(clip_path))
        assert len(frames) == audio_features.shape[0]
        frame_indices = range(len(frames))
        
        # Target frame
        if self.mode == 'train':
            anchor_idx = random.choice(frame_indices) # during training, the target frame is chosen randomly
        elif self.mode == 'sample':
            anchor_idx = 0 # during sampling, the initial target frame matches the first frame of the sequence        
        image = self.preprocess_image(os.path.join(clip_path, frames[anchor_idx]))
                                
        # Identity frame: [0,...,min(t+ms,T)]
        id_idx = random.choice(range(min(len(frames), anchor_idx+self.max_shortcut)))
        if self.mode == 'sample' and self.force_align:
            print('------------ FORCED ALIGNMENT ------------')
            id_idx = 0
        id_frame = self.preprocess_image(os.path.join(clip_path, frames[id_idx]))
                
        # Mask image
        if self.mode == 'train':
            masked_image = image.copy()
            landmarks = self.load_pickle(os.path.join(landmarks_path, frames[anchor_idx].replace('jpg', 'pkl')))
            if landmarks is not None:
                mouth = landmarks[48:68]
                min_x, min_y = int(min(mouth[:, 0]))-5, int(min(mouth[:, 1]))-5
            else:
                landmarks = self.load_pickle('/gpu-data2/jpik/MEAD_v2/mean_landmarks.pkl')
                min_x, min_y = 64, 64
            masked_landmarks = np.clip(landmarks[0:48], 0, 128)
            masked_landmarks = (masked_landmarks/64)-1.0
            masked_image[min_y:,:,:] = -1
        elif self.mode == 'sample':
            all_masked = list()
            all_masked_landmarks = list()
            for k in range(len(frames)):
                img = self.preprocess_image(os.path.join(clip_path, frames[k]))
                masked_img = img.copy()
                landmarks = self.load_pickle(os.path.join(landmarks_path, frames[k].replace('jpg', 'pkl')))
                if landmarks is not None:
                    mouth = landmarks[48:68]
                    min_x, min_y = int(min(mouth[:, 0]))-5, int(min(mouth[:, 1]))-5
                else:
                    landmarks = self.load_pickle('/gpu-data2/jpik/MEAD_v2/mean_landmarks.pkl')
                    min_x, min_y = 64, 64
                masked_landmarks = np.clip(landmarks[0:48], 0, 128)
                masked_landmarks = (masked_landmarks/64)-1.0
                masked_img[min_y:,:,:] = -1
                all_masked.append(masked_img)
                all_masked_landmarks.append(masked_landmarks.ravel())
            
        # Audio
        if self.mode == 'train':
            assert [min(max(anchor_idx+i, 0), len(audio_features)-1) for i in range(-self.audio_window, self.audio_window+1)][self.audio_window] == anchor_idx
            audio = np.array([audio_features[min(max(anchor_idx+i, 0), len(audio_features)-1)] for i in range(-self.audio_window, self.audio_window+1)])
        elif self.mode == 'sample':
            audio = np.array(audio_features)
            assert len(frames) == len(audio_features)
                                
        # Construct output dict
        example["image"] = image
        example["identity"] = id_frame
        if self.mode == 'train':
            example["masked_image"] = masked_image
            example["masked_landmarks"] = masked_landmarks.ravel()
        elif self.mode == 'sample':
            example["masked_image"] = np.stack(all_masked, axis=0)
            example["masked_landmarks"] = np.stack(all_masked_landmarks, axis=0)
        example["audio"] = audio
        example["class_label"] = self.emotion2label[emotion]
        example["human_label"] = emotion
        example["anchor_path"] = os.path.join(clip_path, frames[anchor_idx])
        example["frame_idx"] = anchor_idx
        example["identity_idx"] = id_idx
        example["num_frames"] = len(frames)
        example["subj"] = subj
        example["lvl"] = lvl
        example["nbr"] = nbr
        return example
           
    def __len__(self):
        return len(self.tuples)
    
# DiffTalk, trainbable WAV2VEC2
class MEADBase4(Dataset):
    def __init__(self, size=128, tuples_path=None, 
                 random_crop=False, mode='train', 
                 max_shortcut=30, force_align=False):
        
        assert size is not None and tuples_path is not None
        assert mode in ['train', 'sample']
        self.size = size
        self.max_shortcut = max_shortcut
        self.random_crop = random_crop
        self.force_align = force_align
        self.mode = mode
        self.clip_template = "/gpu-data2/jpik/MEAD/{subj}/video/front/{emotion}/{lvl}/{nbr}"
        self.landmarks_template = "/gpu-data2/jpik/MEAD/{subj}/landmarks/front/{emotion}/{lvl}/{nbr}"
        self.audio_template = "/gpu-data3/filby/MEAD/{subj}/audio/{emotion}/{lvl}/{nbr}.wav"
        self.audio_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_CONFIG)
        self.emotion2label = {"angry": 6, "contempt": 7, "disgusted": 5, "fear": 4, "happy": 1, "neutral": 0, "sad": 2, "surprised": 3}
        self.label2emotion = {v: k for k, v in self.emotion2label.items()}
        
        with open(tuples_path, 'rb') as handle:
            self.tuples = sorted(list(pickle.load(handle)))
            
#         self.noiser = albumentations.augmentations.transforms.GaussNoise(var_limit=1)
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def fill_frames(self, current_idx, substitute_idx):
        if current_idx < 0:
            return substitute_idx
        else:
            return current_idx
    
    def __getitem__(self, idx):
        example = dict()
        subj, emotion, lvl, nbr = self.tuples[idx]
        wav_path = self.audio_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        clip_path = self.clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        landmarks_path = self.landmarks_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        frames = sorted(os.listdir(clip_path))
        frame_indices = range(len(frames))
        
        # target frame
        if self.mode == 'train':
            anchor_idx = random.choice(frame_indices) # during training, the target frame is chosen randomly
        elif self.mode == 'sample':
            anchor_idx = 0 # during sampling, the initial target frame matches the first frame of the sequence
        image = self.preprocess_image(os.path.join(clip_path, frames[anchor_idx]))
        
        # Identity frame: [0,...,min(t+30,T)]
        id_idx = random.choice(range(min(len(frames), anchor_idx+self.max_shortcut)))
        if self.mode == 'sample' and self.force_align:
            print('------------ FORCED ALIGNMENT ------------')
            id_idx = 0
        id_frame = self.preprocess_image(os.path.join(clip_path, frames[id_idx]))

        # Audio
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        audio_feature = np.squeeze(self.audio_processor(speech_array, sampling_rate=16000).input_values)
        
        # Mask image
        if self.mode == 'train':
            masked_image = image.copy()
            landmarks = self.load_pickle(os.path.join(landmarks_path, frames[anchor_idx].replace('jpg', 'pkl')))
            mouth = landmarks[48:68]
            min_x, min_y = int(min(mouth[:, 0])), int(min(mouth[:, 1]))
            masked_image[min_y-5:,:,:] = -1
        elif self.mode == 'sample':
            all_masked = list()
            for k in range(len(frames)):
                img = self.preprocess_image(os.path.join(clip_path, frames[k]))
                masked_img = img.copy()
                landmarks = self.load_pickle(os.path.join(landmarks_path, frames[k].replace('jpg', 'pkl')))
                mouth = landmarks[48:68]
                min_x, min_y = int(min(mouth[:, 0])), int(min(mouth[:, 1]))
                masked_img[min_y-5:,:,:] = -1
                all_masked.append(masked_img)
                                
        # Construct output dict
        example["image"] = image
        example["identity"] = id_frame
        if self.mode == 'train':
            example["masked_image"] = masked_image
        elif self.mode == 'sample':
            example["masked_image"] = np.stack(all_masked, axis=0)
        example["audio"] = audio_feature
        example["class_label"] = self.emotion2label[emotion]
        example["human_label"] = emotion
        example["anchor_path"] = os.path.join(clip_path, frames[anchor_idx])
        example["frame_idx"] = anchor_idx
        example["identity_idx"] = id_idx
        example["num_frames"] = len(frames)
        example["subj"] = subj
        example["lvl"] = lvl
        example["nbr"] = nbr
        return example
           
    def __len__(self):
        return len(self.tuples)
    
    
# DiffTalk, frozen WAV2VEC2    
class MEADBase5(Dataset):
    def __init__(self, audio_window, size=128, tuples_path=None, 
                 random_crop=False, mode='train', 
                 max_shortcut=60, force_align=False):
        
        assert size is not None and tuples_path is not None
        assert mode in ['train', 'sample']
        self.size = size
        self.max_shortcut = max_shortcut
        self.random_crop = random_crop
        self.force_align = force_align
        self.audio_window = audio_window
        self.mode = mode
        self.clip_template = "/gpu-data2/jpik/MEAD_v2/{subj}/video/front/{emotion}/{lvl}/{nbr}"
        self.landmarks_template = "/gpu-data2/jpik/MEAD_v2/{subj}/landmarks/front/{emotion}/{lvl}/{nbr}"
        self.audio_dir = "/gpu-data2/jpik/MEAD/precomputed_audio_features"
        self.emotion2label = {"angry": 6, "contempt": 7, "disgusted": 5, "fear": 4, "happy": 1, "neutral": 0, "sad": 2, "surprised": 3}
        self.label2emotion = {v: k for k, v in self.emotion2label.items()}
        
        with open(tuples_path, 'rb') as handle:
            self.tuples = sorted(list(pickle.load(handle)))
            
        self.noiser = albumentations.augmentations.transforms.GaussNoise(var_limit=1)
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def load_pickle(self, path):
        if os.path.getsize(path) > 0:
            with open(path, 'rb') as handle:
                b = pickle.load(handle)
            return b
        else:
            return None
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def fill_frames(self, current_idx, substitute_idx):
        if current_idx < 0:
            return substitute_idx
        else:
            return current_idx
    
    def __getitem__(self, idx):
        example = dict()
        subj, emotion, lvl, nbr = self.tuples[idx]
        clip_path = self.clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        landmarks_path = self.landmarks_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        audio_path = os.path.join(self.audio_dir, f"{subj}_{emotion}_{lvl}_{nbr}.pkl")
        audio_features = self.load_pickle(audio_path)
        frames = sorted(os.listdir(clip_path))
        assert len(frames) == audio_features.shape[0]
        frame_indices = range(len(frames))
        
        # Target frame
        if self.mode == 'train':
            anchor_idx = random.choice(frame_indices) # during training, the target frame is chosen randomly
        elif self.mode == 'sample':
            anchor_idx = 0 # during sampling, the initial target frame matches the first frame of the sequence        
        image = self.preprocess_image(os.path.join(clip_path, frames[anchor_idx]))
                                
        # Identity frame: [0,...,min(t+ms,T)]
        id_idx = random.choice(range(min(len(frames), anchor_idx+self.max_shortcut)))
        if self.mode == 'sample' and self.force_align:
            print('------------ FORCED ALIGNMENT ------------')
            id_idx = 0
        id_frame = self.preprocess_image(os.path.join(clip_path, frames[id_idx]))
                
        # Mask image
        if self.mode == 'train':
            masked_image = image.copy()
            landmarks = self.load_pickle(os.path.join(landmarks_path, frames[anchor_idx].replace('jpg', 'pkl')))
            if landmarks is not None:
                mouth = landmarks[48:68]
                min_x, min_y = int(min(mouth[:, 0]))-5, int(min(mouth[:, 1]))-5
            else:
                landmarks = self.load_pickle('/gpu-data2/jpik/MEAD_v2/mean_landmarks.pkl')
                min_x, min_y = 64, 64
            masked_landmarks = np.clip(landmarks[0:48], 0, 128)
            masked_landmarks = (masked_landmarks/64)-1.0
            masked_image[min_y:,:,:] = -1
        elif self.mode == 'sample':
            all_masked = list()
            all_masked_landmarks = list()
            for k in range(len(frames)):
                img = self.preprocess_image(os.path.join(clip_path, frames[k]))
                masked_img = img.copy()
                landmarks = self.load_pickle(os.path.join(landmarks_path, frames[k].replace('jpg', 'pkl')))
                if landmarks is not None:
                    mouth = landmarks[48:68]
                    min_x, min_y = int(min(mouth[:, 0]))-5, int(min(mouth[:, 1]))-5
                else:
                    landmarks = self.load_pickle('/gpu-data2/jpik/MEAD_v2/mean_landmarks.pkl')
                    min_x, min_y = 64, 64
                masked_landmarks = np.clip(landmarks[0:48], 0, 128)
                masked_landmarks = (masked_landmarks/64)-1.0
                masked_img[min_y:,:,:] = -1
                all_masked.append(masked_img)
                all_masked_landmarks.append(masked_landmarks.ravel())
            
        # Audio
        if self.mode == 'train':
            assert [min(max(anchor_idx+i, 0), len(audio_features)-1) for i in range(-self.audio_window, self.audio_window+1)][self.audio_window] == anchor_idx
            audio = np.array([audio_features[min(max(anchor_idx+i, 0), len(audio_features)-1)] for i in range(-self.audio_window, self.audio_window+1)])
        elif self.mode == 'sample':
            audio = np.array(audio_features)
            assert len(frames) == len(audio_features)
                                
        # Construct output dict
        example["image"] = image
        example["identity"] = id_frame
        if self.mode == 'train':
            example["landmarks"] = landmarks
            example["masked_landmarks"] = masked_landmarks.ravel()
            example["masked_image"] = masked_image
        elif self.mode == 'sample':
            example["masked_landmarks"] = np.stack(all_masked_landmarks, axis=0)
            example["masked_image"] = np.stack(all_masked, axis=0)
        example["audio"] = audio
        example["class_label"] = self.emotion2label[emotion]
        example["human_label"] = emotion
        example["anchor_path"] = os.path.join(clip_path, frames[anchor_idx])
        example["frame_idx"] = anchor_idx
        example["identity_idx"] = id_idx
        example["num_frames"] = len(frames)
        example["subj"] = subj
        example["lvl"] = lvl
        example["nbr"] = nbr
        return example
           
    def __len__(self):
        return len(self.tuples)
    
    
# DiffTalk, frozen WAV2VEC2    
class MEADBase6(Dataset):
    def __init__(self, audio_window, size=128, tuples_path=None, 
                 random_crop=False, mode='train', 
                 max_shortcut=60, force_align=False):
        
        assert size is not None and tuples_path is not None
        assert mode in ['train', 'sample']
        self.size = size
        self.max_shortcut = max_shortcut
        self.random_crop = random_crop
        self.force_align = force_align
        self.audio_window = audio_window
        self.mode = mode
        self.clip_template = "/gpu-data2/jpik/MEAD_v2/{subj}/video/front/{emotion}/{lvl}/{nbr}"
        self.landmarks_template = "/gpu-data2/jpik/MEAD_v2/{subj}/landmarks/front/{emotion}/{lvl}/{nbr}"
        self.audio_dir = "/gpu-data2/jpik/MEAD/bundle_audio_features"
        self.emotion2label = {"angry": 6, "contempt": 7, "disgusted": 5, "fear": 4, "happy": 1, "neutral": 0, "sad": 2, "surprised": 3}
        self.label2emotion = {v: k for k, v in self.emotion2label.items()}
        
        with open(tuples_path, 'rb') as handle:
            self.tuples = sorted(list(pickle.load(handle)))
            
        self.noiser = albumentations.augmentations.transforms.GaussNoise(var_limit=1)
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def load_pickle(self, path):
        if os.path.getsize(path) > 0:
            with open(path, 'rb') as handle:
                b = pickle.load(handle)
            return b
        else:
            return None
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def fill_frames(self, current_idx, substitute_idx):
        if current_idx < 0:
            return substitute_idx
        else:
            return current_idx
    
    def __getitem__(self, idx):
        example = dict()
        subj, emotion, lvl, nbr = self.tuples[idx]
        clip_path = self.clip_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        landmarks_path = self.landmarks_template.format(subj=subj, emotion=emotion, lvl=lvl, nbr=nbr)
        audio_path = os.path.join(self.audio_dir, f"{subj}_{emotion}_{lvl}_{nbr}.pkl")
        audio_features = self.load_pickle(audio_path)
        frames = sorted(os.listdir(clip_path))
        assert len(frames) == audio_features.shape[0]
        frame_indices = range(len(frames))
        
        # Target frame
        if self.mode == 'train':
            anchor_idx = random.choice(frame_indices) # during training, the target frame is chosen randomly
        elif self.mode == 'sample':
            anchor_idx = 0 # during sampling, the initial target frame matches the first frame of the sequence        
        image = self.preprocess_image(os.path.join(clip_path, frames[anchor_idx]))
                                
        # Identity frame: [0,...,min(t+ms,T)]
        id_idx = random.choice(range(min(len(frames), anchor_idx+self.max_shortcut)))
        if self.mode == 'sample' and self.force_align:
            print('------------ FORCED ALIGNMENT ------------')
            id_idx = 0
        id_frame = self.preprocess_image(os.path.join(clip_path, frames[id_idx]))
                
        # Mask image
        if self.mode == 'train':
            masked_image = image.copy()
            landmarks = self.load_pickle(os.path.join(landmarks_path, frames[anchor_idx].replace('jpg', 'pkl')))
            if landmarks is not None:
                mouth = landmarks[48:68]
                min_x, min_y = int(min(mouth[:, 0]))-5, int(min(mouth[:, 1]))-5
            else:
                landmarks = self.load_pickle('/gpu-data2/jpik/MEAD_v2/mean_landmarks.pkl')
                min_x, min_y = 64, 64
            masked_landmarks = np.clip(landmarks[0:48], 0, 128)
            masked_landmarks = (masked_landmarks/64)-1.0
            masked_image[min_y:,:,:] = -1
        elif self.mode == 'sample':
            all_masked = list()
            all_masked_landmarks = list()
            for k in range(len(frames)):
                img = self.preprocess_image(os.path.join(clip_path, frames[k]))
                masked_img = img.copy()
                landmarks = self.load_pickle(os.path.join(landmarks_path, frames[k].replace('jpg', 'pkl')))
                if landmarks is not None:
                    mouth = landmarks[48:68]
                    min_x, min_y = int(min(mouth[:, 0]))-5, int(min(mouth[:, 1]))-5
                else:
                    landmarks = self.load_pickle('/gpu-data2/jpik/MEAD_v2/mean_landmarks.pkl')
                    min_x, min_y = 64, 64
                masked_landmarks = np.clip(landmarks[0:48], 0, 128)
                masked_landmarks = (masked_landmarks/64)-1.0
                masked_img[min_y:,:,:] = -1
                all_masked.append(masked_img)
                all_masked_landmarks.append(masked_landmarks.ravel())
            
        # Audio
        if self.mode == 'train':
            assert [min(max(anchor_idx+i, 0), len(audio_features)-1) for i in range(-self.audio_window, self.audio_window+1)][self.audio_window] == anchor_idx
            audio = np.array([audio_features[min(max(anchor_idx+i, 0), len(audio_features)-1)] for i in range(-self.audio_window, self.audio_window+1)])
        elif self.mode == 'sample':
            audio = np.array(audio_features)
            assert len(frames) == len(audio_features)
                                
        # Construct output dict
        example["image"] = image
        example["identity"] = id_frame
        if self.mode == 'train':
            example["masked_image"] = masked_image
            example["masked_landmarks"] = masked_landmarks.ravel()
        elif self.mode == 'sample':
            example["masked_image"] = np.stack(all_masked, axis=0)
            example["masked_landmarks"] = np.stack(all_masked_landmarks, axis=0)
        example["audio"] = audio
        example["class_label"] = self.emotion2label[emotion]
        example["human_label"] = emotion
        example["anchor_path"] = os.path.join(clip_path, frames[anchor_idx])
        example["frame_idx"] = anchor_idx
        example["identity_idx"] = id_idx
        example["num_frames"] = len(frames)
        example["subj"] = subj
        example["lvl"] = lvl
        example["nbr"] = nbr
        return example
           
    def __len__(self):
        return len(self.tuples)