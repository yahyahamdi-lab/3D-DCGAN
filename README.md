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

### Generation
Examples of Generated images using Otolith dataset:

<p align="center"><img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Generated_images.PNG" width=450 height=400 /> </p>

### Completion

Example of completed image using Otolith dataset:

<p align="center"><img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completed_images.PNG" width=360 height=220 /> </p>

==> We evaluate the effects of the percentage of removed points on the completion accuracy of 3D objects (Otoliths, Chairs, and Airplanes) using three metrics such as Chamfer Distance (CD), Hausdorff Distance (HD), and Earth Mover’s Distance (EMD).

<p align="center">
<img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completion_pourcentage_cd_metric.PNG" width=550 height=450 /> 
<img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completion_pourcentage_hd_metric.PNG" width=550 height=450 />
<img src="https://github.com/yahyahamdi-lab/3D-DCGAN/blob/main/Completion_pourcentage_EMD_metric.PNG" width=550 height=450 />
</p>

# Tutorial

### Requirements
- Python: Version 3.6 or higher
- PyTorch: Version 1.5 or higher (with CUDA support if a GPU is used).
## Train model
- cd [place_to_clone_this_project]
- [git clone https://github.com/yahyahamdi-lab/3D-DCGAN.git]
- cd 3D-DCGAN
- python Train.py
  
An example of training DCGAN model with Otolith dataset is presented in <a href="https://github.com/yahyahamdi-lab/3D-DCGAN/tree/main/plot"> plot</a> repository with different epochs.
## Evaluate model
- python Test.py
  
# References
[1] A. Jolicoeur-Martineau, The relativistic discriminator: a key element missing from standard GAN, Machine Learning, arXiv:1807.00734v3, 2018.

[2] N. Andrialovanirina, L. Poloni, R. Lafont, E. Poisson Caillault, S. Couette, K. Mahé. 3D meshes dataset of sagittal otoliths from red mullet in the Mediterranean Sea, Sci Data
11, 813, 2024.

[3] Zhang, J., X. Chen, Z. Cai, L. Pan, H. Zhao, S. Yi, C. K. Yeo, B. Dai, and C. C. Loy (2021). Unsupervised 3d shape completion through GAN inversion. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 1768–1777. Computer Vision Foundation / IEEE.

[4] Chang, A. X., T. A. Funkhouser, L. J. Guibas, P. Hanrahan, Q.-X. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi, and F. Yu (2015). Shapenet: An information-rich 3d model repository. ArXiv abs/1512.03012.

[5] Wu, J., C. Zhang, T. Xue, B. Freeman, and J. Tenenbaum (2016). Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In D. D. Lee, M. Sugiyama,
U. von Luxburg, I. Guyon, and R. Garnett (Eds.), Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pp. 82–90.

