from taming.data.custom import MEADBase3
from torch.utils.data import DataLoader
import os, sys
import torch
from omegaconf import OmegaConf
import numpy as np 
from PIL import Image
from einops import rearrange
from ldm.util import instantiate_from_config
from tqdm import tqdm
import pickle

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


# 1. config file
# 2. checkpoint file
# 3. force_align
# 4. steps
# 5. eta
# 6. scale
# 7. num_videos
# 8. postfix
# 9. audio_window

# Input expected to be of type float in the range [0, 1] or [-1, 1]
def rgb2gray(rgb):
    return np.repeat(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])[...,None], 3, 2)


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
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
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
        assert conditioning is not None
        assert len(conditioning.keys()) == 2 
        
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
            e_t = self.model.apply_model(x, t, c['class_label_&_audio'], c['motion_&_id'])
        else:
            e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, c['motion_&_id'])
            e_t = self.model.apply_model(x, t, c['class_label_&_audio'], c['motion_&_id'])
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

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
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def progressive_sampling(self,
                             c1,
                             xid,
                             xmasks,
                             audio_feats,
                             S,
                             batch_size,
                             num_frames,
                             shape,
                             audio_window,
                             eta=0.,
                             verbose=True,
                             unconditional_guidance_scale=1.,
                             unconditional_conditioning=None,
                             **kwargs
                             # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                            ):
        assert c1 is not None
        assert eta in [0., 1.]
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        if len(audio_feats.shape) == 3:
            assert audio_feats.shape[0] == 1
            audio_feats = audio_feats.squeeze(0)

        num_frames = audio_feats.shape[0]
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        device = self.model.betas.device
        timesteps = self.ddim_timesteps

        total_steps = timesteps.shape[0]
        print(f"Running DDIM with {total_steps} timesteps")
        
        zid = xid.clone().detach() # identity latent, [-1, 1], (1, c, h, w)
        generated_frame_latents = list()
        generated_motion_latents = None
        with torch.no_grad():
            for frame_idx in range(num_frames):
                audio_indices = [min(max(frame_idx+i, 0), num_frames-1) for i in range(-audio_window, audio_window+1)]
                assert audio_indices[audio_window] == frame_idx
                print(f'Audio indices: {audio_indices}')
                c2 = audio_feats[audio_indices].unsqueeze(0) # (1, 9, 768)
                c2 = self.model.cond_stage_model_2(c2) # (1, 1, 768)
                c12 = torch.cat([c1, c2], dim=2) # (1, 1, 1024)
                masked_img = xmasks[frame_idx].unsqueeze(0) # masked, RGB, (1, C, H, W)
                assert len(masked_img.shape) == 4 and masked_img.shape[0] == 1
                c3 = self.model.encode_first_stage(masked_img) # masked, latent, (1, c, h, w)
                c34 = torch.cat([c3, zid], dim=1) # (1, 2c, h, w)                
                
                if unconditional_conditioning is not None:
                    uc = torch.cat([unconditional_conditioning, c2], dim=21)
                else:
                    uc = None
                
                c = {'class_label_&_audio': c12, 'motion_&_id': c34}
                img = torch.randn(size, device=device)
                
                iterator = tqdm(np.flip(timesteps), desc=f'Reverse DDIM - Generating frame #{frame_idx}/{num_frames-1}', total=total_steps) 
                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

                    outs = self.p_sample_ddim(x=img, c=c, t=ts, index=index, 
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=uc)
                    img, _ = outs
                
                generated_frame_latents.append(img) # use generated frame latent as the reference/identity latent for the generation of the next frame
                zid = img.clone().detach() # reference latent, [-1, 1], (1, c, h, w)
        
        return generated_frame_latents, generated_motion_latents

    
def main():    
    
    os.makedirs('samples/pickles', exist_ok=True)
    cnfg = str(sys.argv[1]).split('/')[-1].split('.')[0]
    ckpt = str(sys.argv[2]).split('/')[-1].split('.')[:-1]
    ckpt = '.'.join(ckpt)
    print(f'Config: {cnfg}') 
    print(f'Checkpoint: {ckpt}')

    model = get_model(config_file=str(sys.argv[1]), model_path=str(sys.argv[2]))
    sampler = DDIMSampler(model)

    force_align = sys.argv[3].lower() == 'true'
    test_dataset = MEADBase3(size=128, 
                             tuples_path='/gpu-data2/jpik/MEAD/test150.pkl',
                             random_crop=False,
                             audio_window=int(sys.argv[9]),
                             mode='sample',
                             force_align=force_align)
    
    ddim_steps = int(sys.argv[4])
    ddim_eta = float(sys.argv[5])
    scale = float(sys.argv[6]) # for unconditional guidance
    num_videos = min(int(sys.argv[7]), len(test_dataset)) # number of videos to generate
    postfix = str(sys.argv[8])
    audio_window = int(sys.argv[9])
    
    print(f'Postfix: {postfix}')
    bs = 1
    print(f'Number of videos to manipulate: {num_videos}')

    test_loader = DataLoader(test_dataset,
                             batch_size=bs,
                             drop_last=False,
                             shuffle=False,
                             sampler=None,
                             num_workers=4,
                             pin_memory=True)

    video_list = list()
    identity_list = list()
    info_list = list()
    print(f"rendering {bs} examples in {ddim_steps} steps, using s={scale:.2f}, eta={ddim_eta:.2f}.")
    with torch.no_grad():
        with model.ema_scope():
            for batch_idx, batch in enumerate(test_loader):

                if batch_idx >= num_videos:
                    break
                
                # all inputs are normalized in the range [-1, 1]
                print(f'Current batch index: {batch_idx}')
                identity = batch['identity'].permute(0, 3, 1, 2).cuda() # 1 x channel x h x w
                masked_images = batch['masked_image'].squeeze(0).permute(0, 3, 1, 2).cuda() # #frames x channel x h x w
                audio = batch['audio'].squeeze(0).cuda() # 1 x #frames x 768
                assert masked_images.shape[0] == audio.shape[0]
                class_label = batch['class_label'].cpu().numpy()[0]
                emotion = batch["human_label"][0]
                subj = batch["subj"][0]
                lvl = batch["lvl"][0]
                nbr = batch["nbr"][0]
                num_frames = batch['num_frames'].cpu().numpy()[0]
                frame_idx = batch['frame_idx'].cpu().numpy()[0]
                assert frame_idx == 0
                id_idx = batch['identity_idx'].cpu().numpy()[0]
                
                # Unconditional embedding
                if scale > 1.0:    
                    ###################### CFG ######################
                    # Class embedder has 8+1=9 classes, so the embedding index ranges from 0 to 8
                    uc = torch.tensor([model.cond_stage_model_1.n_classes]*bs)[:, None]
                    uc = model.cond_stage_model_1.embedding(uc.to(model.device))
                else:
                    uc = None

                # Conditional Embedding
                c1 = model.cond_stage_model_1.embedding(torch.tensor([class_label]*bs)[:, None].to(model.device)) # label embedding
                zid = model.encode_first_stage(identity) # [-1, 1], identity latent, RGB
#                 print(zid.shape, masked_images.shape, identity.shape, audio.shape)
#                 sys.exit()
                                
                generated_frame_latents, generated_motion_latents = sampler.progressive_sampling(S=ddim_steps,
                                                                                                 c1=c1,
                                                                                                 xid=zid,
                                                                                                 xmasks=masked_images,
                                                                                                 audio_feats=audio,
                                                                                                 batch_size=bs,
                                                                                                 num_frames=num_frames,
                                                                                                 shape=[3, 32, 32],
                                                                                                 verbose=False,
                                                                                                 unconditional_guidance_scale=scale,
                                                                                                 unconditional_conditioning=uc, 
                                                                                                 eta=ddim_eta,
                                                                                                 audio_window=audio_window)
                
                generated_frame_latents = [model.decode_first_stage(x) for x in generated_frame_latents]
                identity_rec = model.decode_first_stage(zid)
                
                generated_frame_latents = [torch.clamp((x+1.0)/2.0, min=0.0, max=1.0) for x in generated_frame_latents]
                identity_rec = torch.clamp((identity_rec+1.0)/2.0, min=0.0, max=1.0)
                
                generated_frame_latents = [x.permute(0, 2, 3, 1).detach().cpu().numpy() for x in generated_frame_latents]
                identity_rec = identity_rec.permute(0, 2, 3, 1).detach().cpu().numpy()
                
                video_list.append(np.concatenate(generated_frame_latents, axis=0))
                identity_list.append(identity_rec)
                
                info_list.append((subj, emotion, lvl, nbr))
                
    with open(f'samples/pickles/video_align={force_align}_nsamples={num_videos}_cnfg={cnfg}_ckpt={ckpt}_steps={ddim_steps}_eta={ddim_eta}_scale={scale}_window={audio_window}_{postfix}.pkl', 'wb') as handle:
        pickle.dump(video_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f'samples/pickles/identity_align={force_align}_nsamples={num_videos}_cnfg={cnfg}_ckpt={ckpt}_steps={ddim_steps}_eta={ddim_eta}_scale={scale}_window={audio_window}_{postfix}.pkl', 'wb') as handle:
        identity_list = np.concatenate(identity_list, axis=0)
        pickle.dump(identity_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f'samples/pickles/info_align={force_align}_nsamples={num_videos}_cnfg={cnfg}_ckpt={ckpt}_steps={ddim_steps}_eta={ddim_eta}_scale={scale}_window={audio_window}_{postfix}.pkl', 'wb') as handle:
        pickle.dump(info_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()