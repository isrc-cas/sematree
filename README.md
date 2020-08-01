# Sematree

The code is based upon [https://github.com/speedinghzl/pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox), and the data processing is based upon [https://github.com/Microsoft/human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch)

### Requirements

python 3.6   

PyTorch 0.4.1  

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.  


Or using anaconda:  conda env create -f environment.yaml  


Or to use Pytorch 1.0, just replace 'libs' with 'modules' in [https://github.com/mapillary/inplace_abn](https://github.com/mapillary/inplace_abn), and rename it to 'libs'. 

Install the Deformable Convolutional Network version 2

### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model
**Note** that the left and right label should be swapped when the label file is flipped. 

Plesae download [LIP](http://sysu-hcp.net/lip/overview.php) dataset and create symbolic links:
ln -s YOUR_LIP_DATASET_DIR dataset/LIP 
  
The contents of LIP Dataset include: 

├── train_images   

├── train_segmentations  

├── val_images  

├── val_segmentations  

├── test_images   

├── train_id.txt  

├── val_id.txt  

├── test_id.txt  

 
Please download imagenet pretrained resent-101 from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into dataset folder.

### Training model
```bash
./run.sh
```

If this code is helpful for your research, please cite the following paper:

    @article{1,
      title={Learning Semantic Neural Tree for Human Parsing},
      author={Ruyi Ji, Dawei Du, Libo Zhang, Longyin Wen, Yanjun Wu, Chen Zhao, Feiyue Huang, Siwei Lyu},
      journal={arXiv:1912.09622},
      year={2019}
    }
