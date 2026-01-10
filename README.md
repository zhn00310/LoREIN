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
### The Architecture Details of CNN in LRR Net:
The specific code of CNN could be found in [folder](/Network/unet)
<p align="center">
  <img src="Figs/2.jpg" alt="Pipeline" width="700"/><br>
  <sub>Fig. 2: Pipeline of CNN in the LRR Net</sub>
</p>
