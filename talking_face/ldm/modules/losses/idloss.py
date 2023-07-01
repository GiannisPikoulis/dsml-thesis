import torch
from torch import nn
from ldm.models.insight_face.model_irse import Backbone, MobileFaceNet
import torchvision.transforms as transforms


MODEL_PATHS = {
    'ir_se50': 'pretrained/model_ir_se50.pth',
}


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.4, mode='ir_se')
        self.facenet.load_state_dict(torch.load(MODEL_PATHS['ir_se50']))
#         self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
#         for param in self.facenet.parameters():
#             param.requires_grad = False
        

    def extract_feats(self, x):
#         if x.shape[2] != 256:
#             x = self.pool(x)
#         x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        # preprocessing assuming samples come in the range [-1, 1], just after getting decoded 
        n_samples = y.shape[0]
        y = torch.clamp((y + 1) * 0.5, min=0, max=1)
        y_hat = torch.clamp((y_hat + 1) * 0.5, min=0, max=1)
        y = self.transform(y)
        y_hat = self.transform(y_hat)
        
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1
        return loss / count