## Diffusion Models with Applications in Face Reenactment and Talking Face Synthesis

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