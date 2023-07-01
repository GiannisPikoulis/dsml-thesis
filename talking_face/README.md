### Preparation
* Download MEAD dataset from [here](https://wywu.github.io/projects/MEAD/MEAD). Please correct all path variables correspondingly.
* Inside the `/data/LRS3_V_WER32.3` directory, place the corresponding `model.pth` and `model.json` files, that can be downloaded from [Model-Zoo](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/tree/master#Model-Zoo). We recommend using the [LSR3 VSR model for multiple languages (32.3 WER)](https://drive.google.com/file/d/1yHd4QwC7K_9Ro2OM_hC7pKUT2URPvm_f/view). If you choose a different model, make sure you edit the `configs/lipread_config.ini` file accordingly.

### First Stage
In order to train first stage autonencoders, please follow the instructions of the [Taming Transformers repository](https://github.com/CompVis/taming-transformers). We recommend using a VQGAN as a first stage model.

### LDM Training
In both face-reenactment and talking-face generation scenarios, LDM training can be performed as follows:
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
```