import argparse
import os

import cv2
import numpy as np
import torch

from backbones import get_model


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class IDLoss(nn.Module):
    def __init__(self, name, weight):
        super(IDLoss, self).__init__()
        self.net = get_model(name, fp16=False)
        self.net.load_state_dict(torch.load(weight))
        self.net.eval()
        self.net.cuda()
        for param in self.facenet.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, y_hat, y):
        bs = y.shape[0]
        y_feats = self.net(y)
        y_hat_feats = self.net(y_hat)
        
        loss = 0
        count = 0
        for i in range(bs):
            sim = y_hat_feats[i].dot(y_feats[i])
            count += 1
        
        return sim / count


def inference(args):
    files = os.listdir(args.dir0)
    sim_index = []
    cnt = 0
    id_loss_func = IDLoss().to(torch.device('cuda')).eval()

    for file in files:
        if(os.path.exists(os.path.join(args.dir1, file))):
            cnt += 1
            
            x = cv2.imread(os.path.join(dir0, file))
            x = cv2.resize(x, (112, 112))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = np.transpose(x, (2, 0, 1))
            x = torch.from_numpy(x).unsqueeze(0).float()
            x.div_(255).sub_(0.5).div_(0.5)
            
            y = cv2.imread(os.path.join(dir1, file))
            y = cv2.resize(y, (112, 112))
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            y = np.transpose(y, (2, 0, 1))
            y = torch.from_numpy(y).unsqueeze(0).float()
            y.div_(255).sub_(0.5).div_(0.5)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            
            # Compute distance
            z = id_loss_func(x, y).item()
            sim_index.append(z)         
        
        else:
            raise ValueError("Given directories don't match")
    
    return np.mean(sim_index)
    
       
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch ArcFace/CosFace Inference')
    parser.add_argument('-d0','--dir0', type=str, required=True) # x0
    parser.add_argument('-d1','--dir1', type=str, required=True) # xrec
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, required=True)
    args = parser.parse_args()
    inference(args)
