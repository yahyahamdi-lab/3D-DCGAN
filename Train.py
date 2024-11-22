
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
import dataIO as data_io
from logger import Logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# hard-wire the GPU ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Argument parser setup
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_workers', type=int, default=2, help='number of data loading workers')
arg_parser.add_argument('--use_cuda', default=0, action='store_true', help='enables cuda')
arg_parser.add_argument('--batchSize', type=int, default=128, help='input batch size')  # default=128
arg_parser.add_argument('--image_dim', type=int, default=64, help='input image dimensions (height/width)')  # default=64
arg_parser.add_argument('--filter_count', type=int, default=64)  # default=64
arg_parser.add_argument('--epochs', type=int, default=500, help='number of epochs')  # default=50
arg_parser.add_argument('--pretrain_epochs', type=int, default=200, help='epochs for generator pretraining')  # default=20
arg_parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
arg_parser.add_argument('--disc_weight', type=float, default=0.001, help='discriminator loss weight')
arg_parser.add_argument('--b1', type=float, default=0.5, help='beta1 for Adam optimizer')
arg_parser.add_argument('--b2', type=float, default=0.99, help='beta2 for Adam optimizer')
arg_parser.add_argument('--seed', type=int, help='manual seed')

options = arg_parser.parse_args()
print(options)

# Set random seed
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)

cudnn.benchmark = True

# Device configuration
device = torch.device("cuda:0" if options.use_cuda else "cpu")
filter_dim = int(options.filter_count)
input_channels = 3
object_class = 'Oto'
object_partial = 'OtoP'
object_ratio = 0.9
local_training = False
batch_counter = 0

# Create necessary directories
if not os.path.exists('logs'):
    os.makedirs('logs')

#logger = Logger('./logs') !!!!!!!!!!!!!!!!!!!!!!!!

if not os.path.exists('models'):
    os.makedirs('models')

model_save_dir = 'home/jerico/Documents/generative-completion/models/'
generated_save_dir = '/home/jerico/Documents/generative-completion/Generated_D/'

graph_dir = model_save_dir + 'plot/'
checkpoint_path = model_save_dir + 'Nets/'

# Helper class to compute and store metrics
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, count=1):
        self.value = value
        self.sum += value * count
        self.count += count
        self.average = self.sum / self.count


class ShapeGenerator(nn.Module):
    def __init__(self):
        super(ShapeGenerator, self).__init__()
        self.network = nn.Sequential(
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

    def forward(self, x):
        return self.network(x)


class ShapeDiscriminator(nn.Module):
    def __init__(self):
        super(ShapeDiscriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
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

        self._initialize_classifier()

    def _initialize_classifier(self):
        dummy_input = torch.zeros(1, 1, options.image_dim, options.image_dim, options.image_dim)
        with torch.no_grad():
            extracted_features = self.feature_extractor(dummy_input)
            print("Feature size: ", extracted_features.size())
            flat_size = extracted_features.size(1) * extracted_features.size(2) * extracted_features.size(3) * extracted_features.size(4)
            print("Flattened size: ", flat_size)
            self.classifier = nn.Sequential(
                nn.Linear(flat_size, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        features = self.feature_extractor(x)
        flattened = features.view(features.size(0), -1)
        output = self.classifier(flattened)
        return output.view(-1, 1).squeeze(1)

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


# Initialize models and weights
generator_model = ShapeGenerator()
weights_init(generator_model)
print(generator_model)

discriminator_model = ShapeDiscriminator()
weights_init(discriminator_model)
print(discriminator_model)


def save_networks(checkpoint_dir, net_g, net_d, epoch):
    print("[*] Saving checkpoints...")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    net_g_name = os.path.join(checkpoint_dir, 'net_g.pth')
    net_d_name = os.path.join(checkpoint_dir, 'net_d.pth')
    net_g_iter_name = os.path.join(checkpoint_dir, f'net_g_{epoch}.pth')
    net_d_iter_name = os.path.join(checkpoint_dir, f'net_d_{epoch}.pth')
    torch.save(net_g.state_dict(), net_g_name)
    torch.save(net_d.state_dict(), net_d_name)
    torch.save(net_g.state_dict(), net_g_iter_name)
    torch.save(net_d.state_dict(), net_d_iter_name)


def save_networks_g(checkpoint_dir, net_g, epoch):
    print("[*] Saving generator checkpoint...")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    net_g_name = os.path.join(checkpoint_dir, 'net_g.pth')
    net_g_iter_name = os.path.join(checkpoint_dir, f'net_g_{epoch}.pth')
    torch.save(net_g.state_dict(), net_g_name)
    torch.save(net_g.state_dict(), net_g_iter_name)


def voxel2points(voxels, threshold=0.3):
    l, m, n = voxels.shape
    X, Y, Z = [], [], []
    positions = np.where(voxels > threshold)
    offpositions = np.where(voxels < threshold)
    voxels[positions] = 1
    voxels[offpositions] = 0
    
    for i, j, k in zip(*positions):
        if np.sum(voxels[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]) < 27:
            X.append(i)
            Y.append(k)
            Z.append(j)
    
    return np.array(X), np.array(Y), np.array(Z)


def voxel2graph(filename, pred, epoch, threshold=0.3):
    X, Y, Z = voxel2points(pred, threshold)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap=cm.viridis, s=25, marker='.')
    plt.title(f'64-3D-DCGAN [Epoch={epoch}]')
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def save_voxels(save_dir, models, epoch):
    print('Saving voxel data...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, str(epoch)), models)

def save_checkpoint(state, curr_epoch):
    torch.save(state, './models/netG_e%d.pth.tar' % (curr_epoch))

# Chargement des voxels (partiels et complets)
partial_voxels = data_io.getAll(obj=object_partial, train=True, is_local=local_training, obj_ratio=object_ratio)
full_voxels = data_io.getAll(obj=object_class, train=True, is_local=local_training, obj_ratio=object_ratio)

# Conversion en tenseurs Torch
partial_voxels = partial_voxels[..., np.newaxis].astype(np.float)
dataP = torch.from_numpy(partial_voxels).permute(0, 4, 1, 2, 3).type(torch.FloatTensor)

full_voxels = full_voxels[..., np.newaxis].astype(np.float)
dataF = torch.from_numpy(full_voxels).permute(0, 4, 1, 2, 3).type(torch.FloatTensor)

# Choix des fonctions de perte
completion_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

# Optimiseurs
optG = optim.Adam(generator_model.parameters(), lr=options.learning_rate, betas=(options.b1,options.b2))
optD = optim.Adam(discriminator_model.parameters(), lr=options.learning_rate, betas=(options.b1, options.b2))


real_label = 1.0
fake_label = 0.0

# Compteurs de perte moyenne
errD_all = MetricTracker()
errG_all = MetricTracker()

# Nombre de batchs
t_batches = int(dataF.size(0) / options.batchSize)

# Boucle principale
for epoch in range(options.epochs):
    epoch_start_time = time.time()
    for i in range(t_batches):
        batch = dataF[i * options.batchSize:(i * options.batchSize + options.batchSize)]
        batchP = dataP[i * options.batchSize:(i * options.batchSize + options.batchSize)]
        
        real_data = batch.to(device)
        partial_voxels = batchP.to(device)

        # **Phase 1 : Entraînement du discriminateur**
        optD.zero_grad()
        label_real = torch.full((options.batchSize,), real_label, device=device).float()
        out_real = discriminator_model(real_data)
        errD_real = adversarial_criterion(out_real, label_real)

        gen_voxels = generator_model(partial_voxels)
        label_fake = torch.full((options.batchSize,), fake_label, device=device).float()
        out_fake = discriminator_model(gen_voxels.detach())
        errD_fake = adversarial_criterion(out_fake, label_fake)

        errD = (errD_real + errD_fake) * 0.5
        errD.backward()
        optD.step()

        # **Phase 2 : Entraînement du générateur**
        optG.zero_grad()
        completion_loss = completion_criterion(gen_voxels, real_data)
        out_fake = discriminator_model(gen_voxels)
        adversarial_loss = adversarial_criterion(out_fake, label_real)

        errG = completion_loss + options.disc_weight * adversarial_loss
        errG.backward()
        optG.step()

        print(f'Epoch [{epoch+1}/{options.epochs}], Batch [{i+1}/{t_batches}] - '
              f'Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f} '
              f'(Completion: {completion_loss.item():.4f}, Adversarial: {adversarial_loss.item():.4f})')

    print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds")

    if (epoch + 1) % 10 == 0:
        save_checkpoint({'epoch': epoch + 1, 'state_dict': discriminator_model.state_dict(), 'optimizer': optG.state_dict()}, epoch + 1)
