import bisect, os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

# EXCLUDE = ['4_4805049ee48f08dc0fb9d58bdd54f576b21e1f86d0af8aa493e63cce.jpg',
#            '4_b2eb8badc9baeee73d91a8a6b3089e5d3f3f6659a7aed13871793b31.jpg',
#            '4_d265fc0f3e75634998020aad227fdafdb6f39b7b626af9268fb8a5ca.jpg',
#            '6_21307084e71799dda2853b5e6663efa287a355e8201b8dd17fb2912e.jpg',
#            '4_8e028edeea264c3d6fb3f1bad97f4e729aa85d2b8fd4f96b5c95aee2.jpg',
#            '4_82f8861ebd4d12b965c30953b450bf0a4968eaf5b440270a86f17d88.jpg',
#            '4_52ce4f5f8a6593e66617d4f532aa2c74aa4c1a884031b6f878c3cdbf.jpg',
#            '4_63f36e0ffa9525c12d083dea4095248108118de55446ae39ba95d1dc.jpg',
#            '6_d2d9b7e8ea4d85c8c390ea39d34eeadd368f50eb141827922a056b38.jpg',
#            '4_85f22e9f900247642097dba92e4e0e445bc81f65cc0dcb9ce87a2c00.jpg',
#            '4_4cecdefdc8ee476f3a6768ab7453ac73b1cdb7746462dcf1a646f749.jpg',
#            '6_c8adf5d39ff1d3ef3b6f6e48d6090ddb9dec4e64e5afdc4023169a91.jpg',
#            '6_eade7a2d4c4da1e706226034250c7b880f8c519aea1d3071449c6795.JPG',
#            '4_43f121f7b412ce0dd5f8c634519691ae4247b634e8ce51a94cce22b7.jpg',
#            '4_473ebfb574ab62a80941d376e6c3e2bed3bb4979045aec7626edec2e.jpeg',
#            '4_45148db593ab0c064a7c206457d5c1f094ff0b79aab80e12ae00be61.jpg',
#            '4_c5f28d2c3fa75be865ebba90bb80e7b05bc6c81783964a8812ea101a.jpg',
#            '6_b7eaffe51399b3b1296a3fb40da7c59ddfabfbaec75ef66bff1c24b6.jpg',
#            '4_261856c8dc53455001d8eb7f533041c0feb1bed7cfc1ae8ae63dc8d6.jpg',
#            '4_47a7ccf29a127c6bcba934be795910b293f253cee8500dcd5b79ead1.png',
#            '4_46fe27cc585483483c3a80d3691b68772bc8795a5dcfcd3831b36372.jpg',
#            '4_b4c1cf39cc90cad7745ba8bf5434663244937526b73c77db64b5727d.jpg',
#            '4_33808a8ec7cda65e92471880090185806d72bcb01d1961ee8008706e.jpg',
#            '6_8781f397dedc7f167424dc09f5b8cfa32bbdbc4d969bf82bca102dfc.png',
#            '4_367bb05a83984246c8eef7d90c4cb8d01b40aa37ec5af42fcbf3617a.jpg',
#            '4_877896bb8b288077c851893e292e5b9a42000486a2737661131e64cb.png',
#            '4_3c82cf3ca3fb057b2a41ff90404a96e786d142ed6ca503b705e8fdda.png',
#            '4_2db23fced9df5fa10a4b877e3d69bb6e9dd2617243d410595d02335c.jpg',
#           ]

EXCLUDE = []


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["file_path_"] = [path for path in self.labels["file_path_"] if path.split('/')[-1] not in set(EXCLUDE)]
        self._length = len(self.labels["file_path_"])

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
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


class AffectnetPaths(ImagePaths):
    def __init__(self, paths, size, random_crop, model, mode):
        super().__init__(paths, size, random_crop)
        self.human_dict = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 
                           4: "fear", 5: "disgust", 6: "anger", 7: "contempt"}
        self.model_name = model
        if self.model_name == 'deca':
            self.shape_path = f"/gpu-data2/jpik/DECA/affectnet_{mode}"
        elif self.model_name == 'emoca':
            self.shape_path = f"/gpu-data2/jpik/emoca/gdl_apps/EMOCA/affectnet_{mode}_aligned_v2"
        else:
            self.shape_path = None
        
    
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        example["class_label"] = int(self.labels["file_path_"][i].split('/')[-1].split('_')[0])
        example["human_label"] = self.human_dict[example["class_label"]]
        example["file_path"] = self.labels["file_path_"][i]
        
        if self.shape_path is not None:
            imagename = self.labels["file_path_"][i].split('/')[-1].split('.')[0]
            if self.model_name == 'emoca':
                shape_image = self.preprocess_image(os.path.join(self.shape_path, imagename, 'geometry_detail.png'))
            else:
                shape_image = self.preprocess_image(os.path.join(self.shape_path, imagename, 'shape_detail_images.jpg'))
            example["shape_image"] = shape_image
        
        return example