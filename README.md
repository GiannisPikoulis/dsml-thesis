## Diffusion Models with Applications in Face Reenactment and Talking-Face Synthesis

### Preparation
* Clone the repo and its submodules:
```
git clone --recurse-submodules -j4 https://github.com/GiannisPikoulis/dsml-thesis
cd dsml-thesis
```
* A suitable conda environment named ldm can be created and activated with:
```
conda env create -f environment.yaml
conda activate ldm
cd talking_face/external/av_hubert/fairseq/
pip install --editable ./
```
### First Stage
In order to train first stage autonencoders, please follow the instructions of the [Taming Transformers repository](https://github.com/CompVis/taming-transformers). We recommend using a VQGAN as a first stage model.

### LDM Training
In both face-reenactment and talking-face generation scenarios, LDM training can be performed as follows:
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
```
### Acknowlegements
* [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
* [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
* [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
* [https://github.com/gwang-kim/DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP)
* [https://github.com/filby89/spectre](https://github.com/filby89/spectre)

### Contact
For questions feel free to open an issue.