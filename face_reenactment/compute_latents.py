from taming.data.custom import AffectnetTrain
from taming.data.custom import AffectnetTest
from torch.utils.data import DataLoader
import os, sys
import torch
from omegaconf import OmegaConf
import numpy as np 
from PIL import Image
from einops import rearrange
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
from tqdm import tqdm
from functools import partial
from collections import Counter


# 1. config file
# 2. checkpoint file
# 3. training data .txt file
# 4. steps
# 5. eta
# 6. scale
# 7. bs
# 8. strength
# 9. mode (train/test)


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
    model, ignored = load_model_from_config(config, model_path, ignore_keys)
    return model


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, strength=1.0, verbose=True):
    if ddim_discr_method == 'uniform':
        #c = num_ddpm_timesteps // num_ddim_timesteps
        #ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        ddim_timesteps = np.linspace(0, 1, num_ddim_timesteps) * int(num_ddpm_timesteps * strength)
        ddim_timesteps = [int(s) for s in list(ddim_timesteps)]
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    
    #steps_out = ddim_timesteps + 1
    steps_out = np.asarray([1] + ddim_timesteps[1:])
    #print(steps_out, len(steps_out))
    
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    #alphas_next = np.asarray(alphacums[ddim_timesteps[1:]].tolist() + [alphacums[-1]])

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev #, alphas_next


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., strength=1.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose, strength=strength)
        alphas_cumprod = self.model.alphas_cumprod
        alphas_cumprod_prev = self.model.alphas_cumprod_prev
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas)) # b[t]
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))  # a[t]
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev)) # a[t-1]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu()))) # sqrt{a[t]}
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu()))) # sqrt{1-a[t]}
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev', to_torch(np.sqrt(1. - alphas_cumprod_prev.cpu()))) # sqrt{1-a[t-1]}
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu()))) # log{1-a[t]}
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu()))) # 1/sqrt{a[t]}
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1))) # sqrt{1/a[t] - 1}

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        self.register_buffer('ddim_sqrt_one_minus_alphas_prev', np.sqrt(1. - ddim_alphas_prev))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):

        b, *_, device = *x.shape, x.device
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

#         if score_corrector is not None:
#             assert self.model.parameterization == "eps"
#             e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        #if quantize_denoised:
        #    pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        #if noise_dropout > 0.:
        #    noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def compute_latents(self,
                        S,
                        batch_size,
                        shape,
                        conditioning=None,
                        callback=None,
                        normals_sequence=None,
                        img_callback=None,
                        quantize_x0=False,
                        eta=0.,
                        mask=None,
                        x0=None,
                        temperature=1.,
                        noise_dropout=0.,
                        score_corrector=None,
                        corrector_kwargs=None,
                        verbose=True,
                        x_T=None,
                        log_every_t=100,
                        unconditional_guidance_scale=1.,
                        unconditional_conditioning=None,
                        strength=0.5,
                        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                        **kwargs
                        ):
        
        #print(unconditional_guidance_scale, unconditional_conditioning)
        assert x0 is not None
        assert conditioning is not None

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, strength=strength)        
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        device = self.model.betas.device
        b = size[0]
        timesteps = self.ddim_timesteps
        print(timesteps)

        total_steps = timesteps.shape[0]
        print(f"Running DDIM with {total_steps} timesteps")

        x = x0.clone().detach()
        iterator = tqdm(timesteps, desc='Forward DDIM', total=total_steps)
        with torch.no_grad():
            for i, step in enumerate(iterator):
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                outs = self.q_sample_ddim(x=x, c=conditioning, t=ts, index=i,
                                          unconditional_guidance_scale=unconditional_guidance_scale, 
                                          unconditional_conditioning=unconditional_conditioning) 
                x, pred_x0 = outs
        
            img = x.clone().detach()
            iterator = tqdm(np.flip(timesteps), desc='Reverse DDIM', total=total_steps)
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                outs = self.p_sample_ddim(x=img, c=conditioning, t=ts, index=index, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
                img, pred_x0 = outs
        
        return img, x, x0

    def q_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        
        b, *_, device = *x.shape, x.device
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            et = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            et_uncond, et = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            et = et_uncond + unconditional_guidance_scale * (et - et_uncond)

#         if score_corrector is not None:
#             assert self.model.parameterization == "eps"
#             et = score_corrector.modify_score(self.model, et, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        #alphas_next = self.ddim_alphas_next
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_alphas_prev = self.model.sqrt_one_minus_alphas_cumprod_prev if use_original_steps else self.ddim_sqrt_one_minus_alphas_prev
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        at = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        at_next = torch.full((b, 1, 1, 1), alphas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas_prev[index], device=device)
        sqrt_one_minus_at_next = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        
        #at = torch.full((b, 1, 1, 1), alphas[index], device=device)
        #at_next = torch.full((b, 1, 1, 1), alphas_next[index], device=device)
        #sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * et) / at.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - at_next).sqrt() * et
        x_next = at_next.sqrt() * pred_x0 + dir_xt 
        
        return x_next, pred_x0
    
    @torch.no_grad()
    def latent_manipulation(self,
                            c_src,
                            c_trg,
                            S,
                            batch_size,
                            shape,
                            callback=None,
                            normals_sequence=None,
                            img_callback=None,
                            quantize_x0=False,
                            eta=0.,
                            mask=None,
                            x0=None,
                            temperature=1.,
                            noise_dropout=0.,
                            score_corrector=None,
                            corrector_kwargs=None,
                            verbose=True,
                            x_T=None,
                            log_every_t=100,
                            unconditional_guidance_scale=1.,
                            unconditional_conditioning=None,
                            strength=1.0,
                            # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                            **kwargs
                            ):
        
        #print(c_trg, unconditional_conditioning)
        assert c_src is not None and c_trg is not None
        assert x0 is not None
        assert x_T is None
        assert eta == 0
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, strength=strength)        
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        device = self.model.betas.device
        b = size[0]
        timesteps = self.ddim_timesteps
        print(timesteps)

        total_steps = timesteps.shape[0]
        print(f"Running DDIM with {total_steps} timesteps")

        x = x0.clone().detach()
        iterator = tqdm(timesteps, desc='Forward DDIM', total=total_steps)
        with torch.no_grad():
            for i, step in enumerate(iterator):
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                outs = self.q_sample_ddim(x=x, c=c_src, t=ts, index=i,
                                          unconditional_guidance_scale=unconditional_guidance_scale, 
                                          unconditional_conditioning=unconditional_conditioning) 
                x, pred_x0 = outs
        
            img = x.clone().detach()
            iterator = tqdm(np.flip(timesteps), desc='Reverse DDIM', total=total_steps)    
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                outs = self.p_sample_ddim(x=img, c=c_trg, t=ts, index=index, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
                img, pred_x0 = outs
        
        return img, x, x0

if __name__ == '__main__':
    cnfg = str(sys.argv[1]).split('/')[-1].split('.')[0]
    ckpt = str(sys.argv[2]).split('/')[-1].split('.')[:-1]
    ckpt = '.'.join(ckpt)
    print(f'Config: {cnfg}') 
    print(f'Checkpoint: {ckpt}')
        
    model = get_model(config_file=str(sys.argv[1]), model_path=str(sys.argv[2]))
    sampler = DDIMSampler(model)

    mode = str(sys.argv[9])
    assert mode in ['train', 'test']

    if mode == 'train':
        assert 'train' in str(sys.argv[3])
        dataset = AffectnetTrain(size=128, 
                                training_images_list_file=f'affectnet/{str(sys.argv[3])}', 
                                model=None, 
                                mode=None)

    elif mode == 'test':
        assert 'val' in str(sys.argv[3])
        dataset = AffectnetTest(size=128, 
                                test_images_list_file=f'affectnet/{str(sys.argv[3])}', 
                                model=None, 
                                mode=None)

    print(f'Number of images to manipulate: {len(dataset)}')

    ddim_steps = int(sys.argv[4])
    ddim_eta = float(sys.argv[5])
    scale = float(sys.argv[6]) # for unconditional guidance
    bs = int(sys.argv[7])
    strength = float(sys.argv[8])

    loader = DataLoader(dataset,
                        batch_size=bs,
                        drop_last=False,
                        shuffle=False,
                        sampler=None,
                        num_workers=4,
                        pin_memory=True)

    if os.path.exists('./samples/precomputed') and os.path.isdir('./samples/precomputed'):
        print("Directory samples/precomputed already exists")
    else:
        os.makedirs('./samples/precomputed', exist_ok=True)

    all_x_lat = list()
    all_x0 = list()
    all_x_rec = list()
    all_file_paths = list()

    device = model.betas.device

    cnt = Counter()
    image_cnt = 0
    THRESHOLD = 5000

    print(f"rendering {bs} examples in {ddim_steps} steps, using s={scale:.2f}, eta={ddim_eta:.2f} and strength={strength:.2f}.")
    with torch.no_grad():
        with model.ema_scope():
            for batch_idx, batch in enumerate(loader):
                
    #             if batch_idx > 4:
    #                 break

                if image_cnt >= THRESHOLD:
                    break
                    
                print(f'Current batch index: {batch_idx} | Images done: {image_cnt}')
        
                img = batch['image'].permute(0, 3, 1, 2).cuda()
                file_paths = batch['file_path']
                class_labels = batch['class_label'].cuda()
                
                # Unconditional embedding
                if scale > 1.0:    
                    ###################### CFG ######################
                    uc = torch.tensor([0]*bs)[:, None]
                    uc = model.cond_stage_model.uncond_embedding(uc.to(model.device))
                else:
                    uc = None
                
                # Conditional Embedding
                c_src = model.cond_stage_model.embedding(class_labels[:, None])

                z = model.encode_first_stage(img)
                x_rec, x_lat, x0 = sampler.compute_latents(S=ddim_steps,
                                                        conditioning=c_src,
                                                        batch_size=bs,
                                                        x0=z,
                                                        shape=[3, 32, 32],
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc, 
                                                        eta=ddim_eta,
                                                        strength=strength)
            
                x0 = model.decode_first_stage(x0)
                x_rec = model.decode_first_stage(x_rec)
            
                x0 = torch.clamp((x0+1.0)/2.0, min=0.0, max=1.0)
                x_rec = torch.clamp((x_rec+1.0)/2.0, min=0.0, max=1.0)

                all_x0.append(x0.permute(0, 2, 3, 1).detach().cpu().numpy())
                all_x_lat.append(x_lat.permute(0, 2, 3, 1).detach().cpu().numpy())
                all_x_rec.append(x_rec.permute(0, 2, 3, 1).detach().cpu().numpy())
                all_file_paths.extend(file_paths)
                cnt += Counter(class_labels.detach().cpu().numpy())
                image_cnt += bs
                
    all_x_lat = np.concatenate(all_x_lat, axis=0) 
    all_x_rec = np.concatenate(all_x_rec, axis=0)
    all_x0 = np.concatenate(all_x0, axis=0)

    THRESHOLD = min(len(dataset), THRESHOLD)

    np.save(f'samples/precomputed/precomputed_{mode}_x0_nsamples{THRESHOLD}_ddim{ddim_steps}_eta{ddim_eta}_scale{scale}_strength{strength}_{cnfg}_{ckpt}.npy', all_x0)
    np.save(f'samples/precomputed/precomputed_{mode}_xrec_nsamples{THRESHOLD}_ddim{ddim_steps}_eta{ddim_eta}_scale{scale}_strength{strength}_{cnfg}_{ckpt}.npy', all_x_rec)   
    np.save(f'samples/precomputed/precomputed_{mode}_xlat_nsamples{THRESHOLD}_ddim{ddim_steps}_eta{ddim_eta}_scale{scale}_strength{strength}_{cnfg}_{ckpt}.npy', all_x_lat)
    np.save(f'samples/precomputed/precomputed_{mode}_fp_nsamples{THRESHOLD}_ddim{ddim_steps}_eta{ddim_eta}_scale{scale}_strength{strength}_{cnfg}_{ckpt}.npy', all_file_paths)

    print(f"Class label counter: {cnt}")