import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import random


class LatentBase(Dataset):
    def __init__(self, size, random_crop, *args, **kwargs):
        super().__init__()
        self.size = size
        self.random_crop = random_crop
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.latents)
    
    def preprocess_image(self, image):        
        image = (255 * image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example['latent'] = self.latents[i]
        example['original'] = self.preprocess_image(self.origin[i])
        if self.fp is not None:
            example['class_label'] = int(self.fp[i].split('/')[-1].split('_')[0])
            example['file_path'] = self.fp[i]
        
        return example

    
class LatentTrain(LatentBase):
    def __init__(self, 
                 training_precomputed_latents_path, 
                 training_origin_path, 
                 training_files_path=None,
                 n_samples=None,
                 size=None, 
                 random_crop=False):
        super().__init__(size, random_crop)
        self.latents = np.load(training_precomputed_latents_path)
        self.origin = np.load(training_origin_path)
        if training_files_path is not None:
            self.fp = np.load(training_files_path)
        else:
            self.fp = None
        
        if n_samples is not None:
            smpl = random.sample(list(enumerate(self.fp)), n_samples)
            smpl_indices = [file[0] for file in smpl]
            self.latents = self.latents[smpl_indices]
            self.origin = self.origin[smpl_indices]
            self.fp = self.fp[smpl_indices]

class LatentTest(LatentBase):
    def __init__(self, 
                 test_precomputed_latents_path, 
                 test_origin_path, 
                 test_files_path=None,
                 n_samples=None,
                 size=None, 
                 random_crop=False):
        super().__init__(size, random_crop)
        self.latents = np.load(test_precomputed_latents_path)
        self.origin = np.load(test_origin_path)
        if test_files_path is not None:
            self.fp = np.load(test_files_path)
        else:
            self.fp = None
            
        if n_samples is not None:
            smpl = random.sample(list(enumerate(self.fp)), n_samples)
            smpl_indices = [file[0] for file in smpl]
            self.latents = self.latents[smpl_indices]
            self.origin = self.origin[smpl_indices]
            self.fp = self.fp[smpl_indices]
