#SAMDNet

by Shengwu Lee(biolee33@gmail.com) at SUST


## Introduction
PyTorch implementation of SAMDNet, which runs at ~2fps with a single CPU core and a single GPU (TITAN XP 12GB).

## Prerequisites
- python 3.6+
- opencv 3.0+
- [PyTorch 1.0+](http://pytorch.org/) and its dependencies 
- for GPU support: a GPU with ~3G memory

## Usage

### Tracking
```bash
 python tracking/run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python tracking/run_tracker.py -s [seq name]```
   - ```python tracking/run_tracker.py -j [json path]```
 
### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Pretraining on ImageNet-VID
   - Download ImageNet-VID dataset into "datasets/ILSVRC"
    ``` bash
     python pretrain/prepro_imagenet.py
     python pretrain/train_mdnet.py -d imagenet
    ```