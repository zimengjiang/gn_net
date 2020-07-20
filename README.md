# 3D Vision: Deep Direct Sparse-to-Dense Localization

## 1. Introduction
### 1.1 Project description:
Visual localization is a key component of many robotics systems. However, in changing conditions such as day-night or summer-winter, it could be very challenging. 
In this project, we push the state-of-the-art in visual localization by building a pipeline which estimates the 6DOF pose of the camera given a query image and a 3D scene. 
To do this, we (1) perform feature-metric PnP given the initial estimation of the pose and the dense features of images, 
(2) train the encoder of Sparse-to-Dense Hypercolumn Matching (S2DHM) method on the supervision of pixel correspondences to generate feature maps and 
(3) integrate feature-metric PnP and learned encoder into the S2DHM framework to construct a visual localization pipeline.

### 1.2 Required dependencies
```
pip install gin-config
pip install torch torchvision
pip install opencv-python
pip install tqdm
pip install -U scikit-learn
pip install pandas
pip install matplotlib
pip install numpy==1.16.1
pip install wandb
pip install plotly
pip install tensorboardX
pip install h5py
pip install imageio
pip install scipy
pip install Pillow
pip install functools
```

### 1.3 Usage:
*Please configure the data root and save root before training! You can change the parameters in run.py.*
```
python run.py
```

### 1.4 Pipeline:
<img src="support_file/img/Selection_010.png" width = 100% height = 100% div align=left />

### 1.5 Useful links:
* [S2DHM](https://github.com/germain-hug/S2DHM)
* [UNET](https://github.com/milesial/Pytorch-UNet)
