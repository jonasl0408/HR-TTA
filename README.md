# HR-OTTA: Robust Online Test-Time Adaptation with Hybrid Fine-Tuning
Official code for [HR-OTTA: Robust Online Test-Time Adaptation with Hybrid Fine-Tuning]

We provide benchmarking and comparison for the following methods:
+ [HR-OTTA]
+ EATA
+ SAR
+ CoTTA
+ TENT
  
on the following tasks
+ CIFAR10/100 -> CIFAR10C/100C
+ ImageNet -> ImageNetC

## Prerequisite
Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate py311
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
## Run
- **Datasets**
  
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. 

The best parameters for each method and dataset are save in ./best_cfgs

download the ckpt of pretrained models and data load sequences from [here](https://drive.google.com/drive/folders/14GWvsEI5pDc3Mm7vqyELeBPuRUSPt-Ao?usp=sharing) and put it in ./ckpt

#### How to reproduce

The entry file to run HR-TTA, TENT, SAR, CoTTA is **test-time-eva.sh**

To evaluate this methods, modify the DATASET and METHOD in test-time-eva.sh

and then

```shell
bash test-time-eva.sh
```

## Acknowledgement 
+ Benchmark-TTA  code is heavily used. [official](https://github.com/yuyongcan/Benchmark-TTA.git)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent) 
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ EATA [official](https://github.com/mr-eggplant/EATA)



## External data link
+ ImageNet-C [Download](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

For questions regarding the code, please contact jixiang.lei@student.tugraz.at.
