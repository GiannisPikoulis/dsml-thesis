import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex, AffectnetPaths


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