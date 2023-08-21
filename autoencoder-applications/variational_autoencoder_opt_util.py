import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pdb
# Shape of MNIST images
image_shape = (28, 28, 1)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 32)
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        t_mean = self.fc_mean(x)
        t_log_var = self.fc_log_var(x)
        return t_mean, t_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 12544)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.view(x.size(0), 64, 14, 14)
        x = torch.relu(self.deconv1(x))
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class SampleFromVariationalDistribution(nn.Module):
    def forward(self, t_mean, t_log_var):
        t_sigma = torch.exp(0.5 * t_log_var)
        epsilon = torch.randn_like(t_mean)
        return t_mean + t_sigma * epsilon

class PredictorLinear(nn.Module):
    def __init__(self, latent_dim):
        super(PredictorLinear, self).__init__()
        
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plot_nll(gx, gy, nll):
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(hspace=0.4)

    for i in range(10):
        plt.subplot(2, 5, i+1)
        gz = nll(i).reshape(gx.shape)
        im = plt.contourf(gx, gy, gz, 
                          cmap='coolwarm', 
                          norm=LogNorm(), 
                          levels=np.logspace(0.2, 1.8, 100))
        plt.title(f'Target = {i}')
    
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, fig.add_axes([0.82, 0.13, 0.02, 0.74]), 
                 ticks=np.logspace(0.2, 1.8, 11), format='%.2f', 
                 label='Negative log likelihood')
    

def train(epoch, model, train_loader, optimizer, 
          log_interval, reconstruction_weight, classification_weight):
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        t_mean, t_log_var, t_decoded, y_pred = model(data)
        
        # Compute losses
        rc_loss = F.binary_cross_entropy_with_logits(
            t_decoded, data.view(-1, 784), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + t_log_var - t_mean.pow(2) - t_log_var.exp())
        ce_loss = F.cross_entropy(y_pred, target)
        
        total_loss = (
            reconstruction_weight * rc_loss +
            kl_loss +
            classification_weight * ce_loss
        )
        
        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch+1} [{batch_idx}/{len(train_loader)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {total_loss.item():.6f}')


def val(model, test_loader, reconstruction_weight, classification_weight):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            t_mean, t_log_var, t_decoded, y_pred = model(data)
            
            # Compute losses
            rc_loss = F.binary_cross_entropy_with_logits(
                t_decoded, data.view(-1, 784), reduction='sum')
            
            kl_loss = -0.5 * torch.sum(1 + t_log_var - t_mean.pow(2) - t_log_var.exp())
            
            ce_loss = F.cross_entropy(y_pred, target)
            
            total_loss = (
                reconstruction_weight * rc_loss +
                kl_loss +
                classification_weight * ce_loss
            )
            
            test_loss += total_loss.item()
            
            # Get the predicted class
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
        f'({100. * correct / len(test_loader.dataset):.2f}%)\n')