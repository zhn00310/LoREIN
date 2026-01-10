# LoREIN
This is the official code for “Unsupervised Highly Accelerated 3D Multi-Parametric MRI Reconstruction via Low-Rank Integrated Implicit Neural Representation”

<p align="center">
  <img src="Figs/1_2.jpg" alt="Pipeline" width="700"/><br>
  <sub>Fig. 1: Overview of the proposed LoREIN framework.</sub>
</p>

## 1. Environmental Requirements
### Some dependencies are essential:
- Python 3.10.X  
- PyTorch 2.0.0
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) ***(Important for neural implicit representation with a multiresolution hash encoding)***
- torchvision, h5py, numpy, scipy, other dependencies

## 2. Detailed Network Structures
### 2.1 The architecture details of CNN in LRR net:
The detailed code of CNN can be found in the folder [unet](/Network/unet)
<p align="center">
  <img src="Figs/2.jpg" alt="Pipeline" width="700"/><br>
  <sub>Fig. 2: Pipeline of CNN in the LRR Net</sub>
</p>

### 2.2 The configuration of hash encoders and MLPs:
The specific settings for hash encoders and MLPs can be found in the file [config_INR.yml](/Network/config_INR.yml)

## 3. Data
### 3.1 The public data of vFA-EPTI: 
The raw k-space data can be downloaded at:
[https://figshare.com/articles/dataset/VFA-EPTI_Datasets/13211669](https://figshare.com/articles/dataset/VFA-EPTI_Datasets/13211669),
and the pre-processed k-space data can be downloaded at:
[https://drive.google.com/drive/folders/1m43pqqx2sqRBQqYfoIiRtNc1wlVhEhGe?usp=drive_link](https://drive.google.com/drive/folders/1m43pqqx2sqRBQqYfoIiRtNc1wlVhEhGe?usp=drive_link)
