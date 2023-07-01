from torch import nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import urllib

CLASS_TO_IDX_7 = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 6}
IDX_TO_CLASS_7 = {6: 'Anger', 5: 'Disgust', 4: 'Fear', 1: 'Happiness', 0: 'Neutral', 2: 'Sadness', 3: 'Surprise'}

CLASS_TO_IDX_8 = {'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7}
IDX_TO_CLASS_8 = {6: 'Anger', 7: 'Contempt', 5: 'Disgust', 4: 'Fear', 1: 'Happiness', 0: 'Neutral', 2: 'Sadness', 3: 'Surprise'}


#def get_path(model_name):
#    return '../../models/affectnet_emotions/' + model_name + '.pt'


def get_model_path(model_name):
    model_file = model_name + '.pt'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotion')
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        url='https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/' + model_file + '?raw=true'
        print('Downloading',model_name,'from',url)
        urllib.request.urlretrieve(url, fpath)
    return fpath        
        

# MODEL_PATHS = {
#     'affectnet-resnet18': '/gpu-data2/jpik/face_model_resnet18.pt',
#     'affectnet-resnet50': '/gpu-data2/jpik/face_model.pt'
# }


# class FaceNet18(nn.Module):
#     def __init__(self):
#         super(FaceNet18, self).__init__()
#         """
#         --> ResNet18 is pre-trained on ImageNet
#         """
#         resnet18 = torchvision.models.resnet18(pretrained = True)
#         self.resnet18_face = nn.Sequential(*list(resnet18.children())[:-1])
#         self.categorical_layer = nn.Linear(in_features=512, out_features=8)
#         self.continuous_layer = nn.Linear(in_features=512, out_features=2)

#     def forward(self, face):
#         x = self.resnet18_face(face).squeeze()
#         if x.shape[0] == 1:
#             x = x.unsqueeze(0)
#         out_cat = self.categorical_layer(x)
#         out_cont = self.continuous_layer(x)
#         return out_cat, out_cont

    
# class FaceNet50(nn.Module):
#     def __init__(self):
#         super(FaceNet50, self).__init__()
#         """
#         --> ResNet18 is pre-trained on ImageNet
#         """
#         resnet50 = torchvision.models.resnet50(pretrained = True)
#         self.resnet50_face = nn.Sequential(*list(resnet50.children())[:-1])
#         self.categorical_layer = nn.Linear(in_features=2048, out_features=8)
#         self.continuous_layer = nn.Linear(in_features=2048, out_features=2)

#     def forward(self, face):
#         x = self.resnet50_face(face).squeeze()
#         if x.shape[0] == 1:
#             x = x.unsqueeze(0)
#         out_cat = self.categorical_layer(x)
#         out_cont = self.continuous_layer(x)
#         return out_cat, out_cont  
    
    
class CLSLoss(nn.Module):
    def __init__(self, model_name, device):
        super(CLSLoss, self).__init__()
        self.device = device
        self.model_name = model_name
        if '_7' in model_name:
            self.idx2class = IDX_TO_CLASS_7
            self.class2idx = CLASS_TO_IDX_7
        else:
            self.idx2class = IDX_TO_CLASS_8
            self.class2idx = CLASS_TO_IDX_8
        path = get_model_path(model_name)
        model = torch.load(path)
        self.model = model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for param in self.model.parameters():
            param.requires_grad = False
        self.loss_fn = torch.nn.CrossEntropyLoss().to(torch.device('cuda')).eval()  
        
    def forward(self, x, target):
        # target: target class index
        # x values are in range [-1, 1] 
        # make x values in range [0, 1]
        # resize to 260x260 for CNN input
        # apply normalization
        x = torch.clamp((x + 1) * 0.5, min=0, max=1)
        x = F.interpolate(x, (260, 260), mode='bilinear')
        x = self.transform(x)
        out = self.model(x)
        trg = torch.LongTensor([ self.class2idx[self.idx2class[target]] ]*x.shape[0]).to(self.device)
        loss = self.loss_fn(out, trg)
        return loss