#SAMDNet

- by Shengwu Lee(biolee3@sina.com) at SUST
- this repository is still under development

## Introduction
PyTorch implementation of SAMDNet, which runs at ~0.76fps with a single CPU core and a single GPU (TITAN XP 12GB).

## Prerequisites
- python 3.6+
- opencv 3.0+
- [PyTorch 1.0+](http://pytorch.org/) and its dependencies 
- for GPU support: a GPU with ~9G memory
- This code is tested on windows 10(1909) 
## Usage

### Online Tracking
 - If you only run the tracker, you can use the pretrained model in models/SAMDNet.pth.
```bash
 python Absolute path of this code/tracking/run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python Absolute path of this code/tracking/run_tracker.py -s [seq name]```
   - ```python Absolute path of this code/tracking/run_tracker.py -j [json path]```
 
### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Pretraining on ImageNet-VID
   - Download ImageNet-VID dataset into "datasets/ILSVRC"
    ``` bash
     python Absolute path of this code/pretrain/prepro_imagenet.py
     python Absolute path of this code/pretrain/train_mdnet.py -d imagenet
    ```
