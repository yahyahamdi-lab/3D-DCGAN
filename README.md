# 3D-DCGAN (Shape generation and completion)

# Methods
This code implements deep 3D Generative Adversarial Network (GAN)-based convolutional layers that operate in voxel space and train to generate or complete 3D voxel-based objects. This work is inspired by the study in https://arxiv.org/abs/1807.00734 [1], with some modifications to the GAN architecture.
## Architecture

![image](https://github.com/user-attachments/assets/62945467-d054-4d52-b369-1d9c0d79f1c8)

## Data
We use:
- Chairs and airplane from Shapenet dataset: https://github.com/yuchenrao/PatchComplete/tree/main?tab=readme-ov-file#download-processed-datasets.
- Otolith dataset from [2].
## Results
<img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completion_pourcentage_cd_metric.PNG" width=350 height=350 />
<img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completion_pourcentage_hd_metric.PNG" width=350 height=350 />
<img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completion_pourcentage_EMD_metric.PNG" width=350 height=350 />

# Tutorial
## Requirements
- Python: Version 3.6 or higher
- PyTorch: Version 1.5 or higher (with CUDA support if a GPU is used).
## Build environment

# References
[1] A. Jolicoeur-Martineau, The relativistic discriminator: a key element missing from standard GAN, Machine Learning, arXiv:1807.00734v3, 2018.

[2] N. Andrialovanirina, L. Poloni, R. Lafont, E. Poisson Caillault, S. Couette, K. Mahé. 3D meshes dataset of sagittal otoliths from red mullet in the Mediterranean Sea, Sci Data
11, 813, 2024.

