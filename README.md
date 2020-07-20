# 3D Vision: Deep Direct Sparse-to-Dense Localization

## 1. Introduction
### 1.1 Project description:
Visual localization is a key component of many robotics systems. However, in changing conditions such as day-night or summer-winter, it could be very challenging. 
In this project, we push the state-of-the-art in visual localization by building a pipeline which estimates the 6DOF pose of the camera given a query image and a 3D scene. 
To do this, we (1) perform feature-metric PnP given the initial estimation of the pose and the dense features of images, 
(2) train the encoder of Sparse-to-Dense Hypercolumn Matching (S2DHM) method on the supervision of pixel correspondences to generate feature maps and 
(3) integrate feature-metric PnP and learned encoder into the S2DHM framework to construct a visual localization pipeline.

### 1.2 Data
You need to follow directory structure of the `data` as below.
```
${gn_net root}
├── data
├── ├── robotcar
|   `── ├── correspondence/
|       |   ├── *.mat
|       ├── images
│           ├── overcast-reference/
│           ├── overcast-summer/
│           ├── overcast-winter/
│           ├── sun/
│           ├── dawn/
│           ├── rain/
│           ├── snow/
│           ├── dusk/
│           ├── night/
│           └── night-rain/
```

### 1.3 Usage:

Run the following commands to install this repository and the required dependencies:

```bash
git clone https://github.com/zimengjiang/gn_net.git
cd gn_net/
pip3 install -r requirements.txt
```
This code was run and tested on Python 3.7.3, using Pytorch 1.5.1 although it should be compatible with some previous versions. You can follow instructions to install Pytorch [here](https://pytorch.org/). *Please configure the data root and save root before training! You can change the parameters in run.py.*
```
python run.py
```

### 1.4 Pipeline:
<img src="support_file/img/pipeline.png" width = 100% height = 100% div align=left />

### 1.5 Useful links:
* [S2DHM](https://github.com/germain-hug/S2DHM)
* [UNET](https://github.com/milesial/Pytorch-UNet)
