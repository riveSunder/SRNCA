import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models

import numpy as np

import skimage
import skimage.io as sio
import skimage.transform

identity = torch.tensor([[0., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 0.]])
sobel_h = torch.tensor([[-1., -1., -1.],
                        [0., 0., 0.],
                        [1., 1., 1.]])
sobel_w = torch.tensor([[-1., 0., 1.],
                        [-1., 0., 1.],
                        [-1., 0., 1.]])
moore = torch.tensor([[1., 1., 1.],
                      [1., 0., 1.] ,
                      [1., 1., 1.]])
laplacian = torch.tensor([[1., 2., 1.],
                          [2., -12., 2],
                          [1., 2., 1.]])

def perceive(x, filters):
    
    batch, channels, height, width = x.shape
    
    x = x.reshape(batch*channels, 1, height, width)
    x = F.pad(x, (1,1,1,1), mode="circular")
    
    x = F.conv2d(x, filters[:, np.newaxis, :, :])
    
    perception = x.reshape(batch, -1, height, width)
    
    return perception

# Parameters for soft clamp from chakazul
soft_clamp = lambda x: 1.0 / (1.0 + torch.exp(-4.0 * (x-0.5)))



class NCA(nn.Module):

    def __init__(self, number_channels=1, number_filters=5, number_hidden=32):
        super().__init__()

        self.number_channels = number_channels
        self.number_filters = number_filters
        self.number_hidden = number_hidden


        self.conv_0 = nn.Conv2d(self.number_channels * self.number_filters, \
                self.number_hidden, kernel_size=1)
        self.conv_1 = nn.Conv2d(self.number_hidden, self.number_channels, \
                kernel_size=1, bias=False)
        self.filters = torch.stack([identity, sobel_h, sobel_w, \
                moore, laplacian])

        self.conv_1.weight.data.zero_()

        self.dt = 1.0
        self.max_value = 1.0
        self.min_value = 0.0

        self.squash = soft_clamp


    def forward(self, grid, update_rate=0.5):

        update_mask = (torch.rand_like(grid) < update_rate) * 1.0
        perception = perceive(grid, self.filters)

        new_grid = self.conv_0(perception)
        new_grid = self.conv_1(new_grid)
        
        new_grid = grid + self.dt * new_grid * update_mask

        return self.squash(new_grid)

    def get_init_grid(self, batch_size=8, dim=128):
        
        temp = torch.zeros(batch_size, self.number_channels, dim, dim)

        return temp

    def count_parameters(self):

        number_parameters = 0

        for param in self.parameters():
            number_parameters += param.numel()

        return number_parameters


    def save_parameters(self, save_path):

        pass

    def load_parameters(self, load_path):

        pass
        
if __name__ == "__main__":

    nca = NCA()

    grid = nca.get_init_grid()

    for step in range(10):
        grid = nca(grid)

    print("OK")

