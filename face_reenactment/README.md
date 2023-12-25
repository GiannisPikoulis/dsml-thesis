## Face-Reenactment
### Preparation
* Download AffectNet dataset from [here](http://mohammadmahoor.com/affectnet/). Please correct all path variables correspondingly.
* In addition, to use ID loss during CLIP-guided finetuning, you are required to download the pretrained [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) model from [TreB1eN](https://github.com/TreB1eN) and put it in the `/pretrained` directory.
### Checkpoints
As requested, we provide snapshots of both the LDM and VQGAN, as trained on the AffectNet dataset. Checkpoint files can be found in this [GDrive folder](https://drive.google.com/drive/folders/1qjzIprDXHqovFWT4GEgRF5OXiM--_4jC?usp=drive_link). We have also included the corresponding PL .yaml configuration files. CLIP-finetuned LDM models will only be released upon request.