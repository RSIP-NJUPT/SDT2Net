# SDT<sup>2</sup>Net
This a official Pytorch implementation of our paper "[Remote Sensing Scene Classification via Second-order Differentiable Token Transformer Network](https://)"


## What SDT<sup>2</sup>Net Does
![pipline](figures/SDT2Net.png)
Abstract...

## Requirements
```
- python >= 3.8
- pytorch >= 1.12.1  
- torchvision        
- timm == 0.4.5      
```


## Data Preparation
- The NWPU-RESISC45 dataset should be prepared as follows:
```
NWPU-RESISC45
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```

## Pre-Trained Models

Users can load the pretrained model we trained in this article to make fine-tuning. Of course, our proposed SDT<sup>2</sup>Net can be operated utilizing the pre-trained models of DeiT. To facilitate seamless integration, our code is programmed to automatically download and load these pre-trained models. However, users who prefer manual downloads can acquire the pre-trained DeiT models through this [link](https://github.com/facebookresearch/deit/blob/main/README_deit.md).

 


## Evaluation
We provide the discovered compression rates in the [compression_rate.json](https://github.com/anonymous998899/DiffRate/blob/main/compression_rate.json) file. To evaluate these rates, utilize the `--load_compression_rate` option, which will load the appropriate compression rate from [compression_rate.json](https://github.com/anonymous998899/DiffRate/blob/main/compression_rate.json) based on the specified `model` and `target_flops`. An example evaluating the `DeiT-S` model with `4.5G` FLOPs would be:
```
python main.py --eval --load_compression_rate --data-path $path_to_imagenet$ --model vit_deit_small_patch16_224 --target_flops 4.5
```

## Training

To find the optimal compression rate by proposed SDT<sup>2</sup>Net, run the following code and search a `4.5G` compression rate schedule for `DeiT-S`:
```
python main.py \
--arch-lr 0.01 --arch-min-lr 0.01 \
--epoch 100 --batch-size 64 \
--data-path $path_to_imagenet$ \
--output_dir $path_to_save_log$ \
--model vit_deit_small_patch16_224 \
--target_flops 4.5
```
- supported `$model_name$`: `{vit_deit_tiny_patch16_224,vit_deit_small_patch16_224,vit_deit_base_patch16_224}`
- supported `$target_flops$`: a floating point number

## Citation
If you use SDT<sup>2</sup>Net or this repository in your work, please cite:
```
@article{,
  title={},
  author={},
  journal={},
  year={2024}
}
```

## Acknowledge
This codebase borrow some code from [DeiT](https://github.com/facebookresearch/deit). Thanks for their wonderful work.
