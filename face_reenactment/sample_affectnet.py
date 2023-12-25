import torch
from omegaconf import OmegaConf
import sys, os
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# 1. config
# 2. ckpt_path
# 3. samples per class
# 4. steps
# 5. eta
# 6. scale
# 7. batch_size
# 8. postfix

def load_model_from_config(config, ckpt, ignore_keys=list()):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt) #, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    ignored = list()
    for k in sd.keys():
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                ignored.append({k: sd[k]})
                del sd[k]
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model, ignored


def get_model(config_file, model_path):  
    config = OmegaConf.load(config_file) 
    ignore_keys = list()
    # if 'v2' in config_file:
    #    print('v2 detected in config filename')
    #    ignore_keys.append('cond_stage_model.embedding.weight')
    
    model, ignored = load_model_from_config(config, model_path, ignore_keys)

    # if 'v2' in config_file:
    #    print('v2 detected in config filename')
    #    print(model.cond_stage_model.embedding.weight.data)
    #    nul = model.cond_stage_model.embedding.weight[8].data.unsqueeze(0)
    #    res = torch.cat([ignored[0]['cond_stage_model.embedding.weight'], nul], 0)
    #    model.cond_stage_model.embedding = torch.nn.Embedding.from_pretrained(res)
    #    print(model.cond_stage_model.embedding.weight.data)
    #    print(model.cond_stage_model.uncond_embedding.weight.data)
    return model

if __name__ == "__main__":
    cnfg = str(sys.argv[1]).split('/')[-1].split('.')[0]
    ckpt = str(sys.argv[2]).split('/')[-1].split('.')[:-1]
    ckpt = '.'.join(ckpt)
    print(f'Config: {cnfg}') 
    print(f'Checkpoint: {ckpt}')

    model = get_model(str(sys.argv[1]), str(sys.argv[2]))
    sampler = DDIMSampler(model)
    print(model.cond_stage_model.embedding.weight.data)

    n_classes = 8
    classes = list(range(n_classes)) # define classes to be sampled here
    n_samples_per_class = int(sys.argv[3])

    print('Total number of samples to generate per class: {}'.format(n_samples_per_class))

    ddim_steps = int(sys.argv[4])
    ddim_eta = float(sys.argv[5])
    scale = float(sys.argv[6])  # for unconditional guidance
    bs = int(sys.argv[7])
    postfix = str(sys.argv[8])
    print(f'ddim_steps: {ddim_steps}, ddim_eta: {ddim_eta}, scale: {scale}')

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            if scale > 1.0:    
                ##################### NO CFG ####################
                #uc = model.get_learned_conditioning(
                #    {model.cond_stage_key: torch.tensor(bs*[8]).to(model.device)}
                #)
                ###################### CFG ######################
                uc = torch.tensor([0]*bs)[:, None]
                uc = model.cond_stage_model.uncond_embedding(uc.to(model.device))
            else:
                uc = None
            
            for class_label in classes:
                cnt = 0
                class_samples = list()
                print(f"Rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                while cnt < n_samples_per_class:
                    ##################### NO CFG #####################               
                    #xc = torch.tensor(bs*[class_label])
                    #print(xc, xc.shape)
                    #c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                    ###################### CFG ######################
                    xc = torch.tensor(bs*[class_label])[:, None]
                    c = model.cond_stage_model.embedding(xc.to(model.device))
                    
                    print(c.shape)
                    if uc is not None:
                        print(uc.shape)
                    else:
                        print(uc)
                
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=bs,
                                                    shape=[3, 32, 32],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                    class_samples.append(x_samples_ddim)
                    cnt += bs
                    print('Total number of generated samples for class {}: {}'.format(class_label, cnt))
                    
                class_samples = torch.cat(class_samples, dim=0).permute(0, 2, 3, 1).cpu().numpy()
                os.makedirs(f'samples/affectnet_conditional/{class_label}', exist_ok=True)
                np.save(f'samples/affectnet_conditional/{class_label}/affectnet_ema_nsamples{n_samples_per_class}_ddim{ddim_steps}_eta{ddim_eta}_scale{scale}_{postfix}_{cnfg}_{ckpt}_emotion={class_label}.npy', class_samples)
                all_samples.append(class_samples)

    np.save(f'samples/affectnet_conditional/affectnet_ema_nsamples{n_samples_per_class*n_classes}_ddim{ddim_steps}_eta{ddim_eta}_scale{scale}_{postfix}_{cnfg}_{ckpt}.npy', all_samples)
