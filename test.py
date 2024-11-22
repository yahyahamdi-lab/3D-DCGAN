
from __future__ import print_function
import argparse
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import dataIO as d
import visdom
from logger import Logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tensorlayer as tl
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh


batchSize = 32
imageSize = 64
localSize = 6
hole_min= 2
hole_max= 4
cuda=0
device = torch.device("cuda:0" if cuda else "cpu")
ob = 'Oto'
obj_ratio = 0.2
is_local = False
batch_index = 0


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=8, dilation=8, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv3d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )

        # Calculate the size of the input to the linear layer
        self._init_linear_layer()

    def _init_linear_layer(self):
        # Use a dummy input to determine the size of the tensor after convolutions
        dummy_input = torch.zeros(1, 1, imageSize, imageSize, imageSize)
        with torch.no_grad():
            x = self.disc(dummy_input)
            print("Size after convolutions: ", x.size())
            n_size = x.size(1) * x.size(2) * x.size(3) * x.size(4)
            print("Flattened size: ", n_size)
            self.classifier = nn.Sequential(
                nn.Linear(n_size, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.disc(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 1).squeeze(1)


# weight initialization
def weights_init(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


#generate "cuboid" noise
def get_points():
    points = []
    mask = []
    for i in range(batchSize):
        x1, y1, z1 = np.random.randint(0, imageSize - localSize + 1, 3)
        x2, y2, z2 = np.array([x1, y1, z1]) + localSize
        points.append([x1, y1, x2, y2, z1, z2])

        w, h, d = np.random.randint(hole_min, hole_max + 1, 3)
        p1 = x1 + np.random.randint(0, localSize - w)
        q1 = y1 + np.random.randint(0, localSize - h)
        r1 = z1 + np.random.randint(0, localSize - d)
        p2 = p1 + w
        q2 = q1 + h
        r2 = r1 + d

        m = np.zeros((1, imageSize, imageSize, imageSize), dtype=np.uint8)
        m[:, q1:q2 + 1, p1:p2 + 1, r1:r2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)

def save_networks_2(checkpoint_dir, sess, net_g, net_d, epoch):
	print("[*] Saving checkpoints...")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	# this saves as the latest version location
	net_g_name = os.path.join(checkpoint_dir, 'net_g.npz')
	net_d_name = os.path.join(checkpoint_dir, 'net_d.npz')
	# this saves as a backlog of models
	net_g_iter_name = os.path.join(checkpoint_dir, 'net_g_%d.npz' % epoch)
	net_d_iter_name = os.path.join(checkpoint_dir, 'net_d_%d.npz' % epoch)
	tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
	tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
	tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
	tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)

def save_networks(checkpoint_dir, net_g, net_d, epoch):
    print("[*] Saving checkpoints...")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # this saves as the latest version location
    net_g_name = os.path.join(checkpoint_dir, 'net_g.pth')
    net_d_name = os.path.join(checkpoint_dir, 'net_d.pth')
    # this saves as a backlog of models
    net_g_iter_name = os.path.join(checkpoint_dir, 'net_g_%d.pth' % epoch)
    net_d_iter_name = os.path.join(checkpoint_dir, 'net_d_%d.pth' % epoch)
    # save the models
    torch.save(net_g.state_dict(), net_g_name)
    torch.save(net_d.state_dict(), net_d_name)
    torch.save(net_g.state_dict(), net_g_iter_name)
    torch.save(net_d.state_dict(), net_d_iter_name)

def voxel2points(voxels, threshold=.3):
	l, m, n = voxels.shape
	X = []
	Y = []
	Z = []
	positions = np.where(voxels > threshold) # recieves position of all voxels
	offpositions = np.where(voxels < threshold) # recieves position of all voxels
	voxels[positions] = 1 # sets all voxels values to 1 
	voxels[offpositions] = 0 
	
	for i,j,k in zip(*positions):
		if np.sum(voxels[i-1:i+2,j-1:j+2,k-1:k+2])< 27 : #identifies if current voxels has an exposed face 
			X.append(i)
			Y.append(k)
			Z.append(j)
	
	return np.array(X),np.array(Y),np.array(Z)

def voxel2graph(filename, pred, epoch, threshold=.3):
	X,Y,Z = voxel2points(pred, threshold)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(X, Y, Z, c=Z, cmap=cm.viridis, s=25, marker='.')
	plt.title('64-3D-RaSGAN [Epoch=%i]' % (epoch))
	plt.savefig(filename, bbox_inches='tight')
	plt.close('all')
     
def save_voxels(save_dir, models, epock): 
	print('Saving the model')
	global batch_index
	batch_index += 1
	if(batch_index >= batchSize):
		batch_index = 0
	#save only one from batch per epoch to save space
	np.save(save_dir+str(epock)  , models)
    #np.save(save_dir+str(epock)  , models[batch_index])
     
def save_checkpoint(state, curr_epoch):
    torch.save(state, './models/netG_e%d.pth.tar' % (curr_epoch))

def test_model(generator, test_data, batch_size, device):
    generator.eval()  # Set the generator to evaluation mode

    test_batches = int(test_data.size(0) / batch_size)
    results = []
    for i in range(test_batches):
        with torch.no_grad():  # Disable gradient calculations for testing
            batch = test_data[i*batch_size:(i*batch_size + batch_size)].to(device)

            # Create masks and apply them to the data
            _, mask_batch = get_points()  # Generate a mask for the test batch
            masks = torch.from_numpy(mask_batch).type(torch.FloatTensor).to(device)
            masked_data = batch * masks  # Apply mask

            # Generate data with the Generator
            generated_data = generator(masked_data)

            # Collect results for analysis
            results.append(generated_data.cpu().numpy())
    
    # Concatenate results for final output
    return np.concatenate(results, axis=0)

def visualize_results_2(test_outputs):
     for k in range(2):#k, voxels_data in enumerate(test_outputs):
          voxels = (test_outputs[0] > 0.5).astype(np.uint8)
          voxels = voxels.squeeze()
          print("Converted voxel data shape:", voxels.dtype) 
          # Step 3: Convert smoothed voxels to a mesh using the Marching Cubes algorithm
          verts, faces, normals, values =  measure.marching_cubes_lewiner(voxels, level=0.5)

          # Step 4: Save the mesh as an STL file
          verts_faces = verts[faces]
          mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
          for i, f in enumerate(faces):
               for j in range(3):
                    mesh_data['vectors'][i][j] = verts_faces[i][j]
          #voxel_mesh = mesh.Mesh(mesh_data)
          fig = plt.figure(figsize=(10, 10))
          ax = fig.add_subplot(111, projection='3d')
          mesh_plot = Poly3DCollection(verts[faces], alpha=0.7, facecolor='red', edgecolor='black')
          ax.add_collection3d(mesh_plot)
          ax.set_xlim(0, 60)
          ax.set_ylim(0, 60)
          ax.set_zlim(0, 60)
          plt.show()


def visualize_results(test_outputs, epoch, threshold=0.5):
    for i, voxels in enumerate(test_outputs):
        # Convert voxel data to points for 3D plotting
        X, Y, Z = voxel2points(voxels.squeeze(), threshold=threshold)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=Z, cmap=cm.viridis, s=25, marker='.')
        plt.title(f'Test Sample {i} - Epoch {epoch}')
        plt.show()


def convert_to_voxels(test_outputs, threshold=0.5):
    """
    Converts the model outputs to a voxel grid by applying a threshold.
    
    Parameters:
    - test_outputs (torch.Tensor or numpy array): The output of the model, expected shape [batch_size, channels, depth, height, width].
    - threshold (float): Threshold to apply for binary conversion of voxel grid.
    
    Returns:
    - voxels (numpy array): Binary voxel grid in shape [batch_size, depth, height, width].
    """
    
    # If test_outputs is a PyTorch tensor, move it to CPU and convert to numpy
    if isinstance(test_outputs, torch.Tensor):
        test_outputs = test_outputs.detach().cpu().numpy()
    
    # Apply thresholding to create binary voxel data
    voxels = (test_outputs > threshold).astype(np.uint8)
    
    # If there's a channel dimension (e.g., [batch_size, 1, D, H, W]), squeeze it out
    if voxels.shape[1] == 1:
        voxels = voxels.squeeze(1)
    
    return voxels

# Load the test dataset (for example, a holdout set)
t_data = d.getAll(obj=ob, train=True, is_local=is_local, obj_ratio=obj_ratio)
t_data = t_data[..., np.newaxis].astype(np.float)
test_data = torch.from_numpy(t_data)
test_data = test_data.permute(0, 4, 1, 2, 3)
test_data = test_data.type(torch.FloatTensor)


# Assuming `Generator` and `Discriminator` classes are already defined
netG = Generator()
netD = Discriminator()
epoch = 80
# Load the .pth file for each model (update path as needed)
netG.load_state_dict(torch.load('E:/Oto3D/generative-3D-mesh-completion-master/new project/Chairs/net_g_20.pth'))
#netG.load_state_dict(torch.load('E:/Oto3D/generative-3D-mesh-completion-master/models_t2/Nets/net_g_200.pth'))
#netD.load_state_dict(torch.load('E:/Oto3D/generative-3D-mesh-completion-master/models_t2/Nets/net_d_10.pth'))
# Run the test model on test data
test_outputs = test_model(netG, test_data, batchSize, device)

# Visualize a few outputs
#visualize_results(test_outputs, epoch)
visualize_results_2(test_outputs)
