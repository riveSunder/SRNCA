import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models

import numpy as np

import skimage
import skimage.io as sio
import skimage.transform

from srnca.utils import compute_grams, perceive, compute_style_loss

identity = torch.tensor([[0., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 0.]])
laplacian = torch.tensor([[1., 2., 1.],
                          [2., -12., 2],
                          [1., 2., 1.]])
sobel_h = torch.tensor([[-1., -1., -1.],
                        [0., 0., 0.],
                        [1., 1., 1.]])
sobel_w = torch.tensor([[-1., 0., 1.],
                        [-1., 0., 1.],
                        [-1., 0., 1.]])
moore = torch.tensor([[1., 1., 1.],
                      [1., 0., 1.] ,
                      [1., 1., 1.]])

# Parameters for soft clamp from chakazul
soft_clamp = lambda x: 1.0 / (1.0 + torch.exp(-4.0 * (x-0.5)))

class NCA(nn.Module):

    def __init__(self, number_channels=1, number_filters=5, number_hidden=32, device="cpu"):
        super().__init__()

        self.number_channels = number_channels
        self.number_hidden = number_hidden

        self.my_device = torch.device(device)

        self.filters = torch.stack([identity, laplacian, sobel_h, sobel_w, \
                moore][:min([5, number_filters])])

        # max filters is 20
        number_filters = min([number_filters, 20])
        while self.filters.shape[0] < number_filters:
            self.filters = torch.cat([self.filters, torch.randn_like(self.filters[0:1])])

        self.number_filters = self.filters.shape[0]
        self.conv_0 = nn.Conv2d(self.number_channels * self.number_filters, \
                self.number_hidden, kernel_size=1)
        self.conv_1 = nn.Conv2d(self.number_hidden, self.number_channels, \
                kernel_size=1, bias=False)

        self.conv_1.weight.data.zero_()

        self.to_device(self.my_device)

        self.dt = 1.0
        self.max_value = 1.0
        self.min_value = 0.0

        self.update_rate = 0.9

        self.squash = soft_clamp


    def forward(self, grid):
    
        update_mask = (torch.rand_like(grid, device=self.my_device) < self.update_rate) * 1.0

        perception = perceive(grid, self.filters)

        new_grid = self.conv_0(perception)
        new_grid = self.conv_1(new_grid)
        
        new_grid = grid + self.dt * new_grid * update_mask

        return self.squash(new_grid)

    def get_init_grid(self, batch_size=8, dim=128):
        
        temp = torch.zeros(batch_size, self.number_channels, dim, dim, \
                device=self.my_device)
        #temp = temp.to(self.my_device)

        return temp

    def initialize_optimizer(self, lr, max_steps):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(\
                self.optimizer, [max_steps // 3], 0.3)

    def save_log(self, my_training, exp_tag=""):
        
        exp_tag += "_" + str(int(time.time()))

        np.save(f"{exp_tag}_log_dict.npy", my_training)
        self.save_parameters(save_path = f"{exp_tag}_model.pt")

        return exp_tag

    def fit(self, target, max_steps=10, lr=1e-3, max_ca_steps=16, batch_size=8, exp_tag="default_tag"):

        self.batch_size = batch_size
        display_every = max_steps // 8 + 1

        self.initialize_optimizer(lr, max_steps)
        target = target.to(self.my_device)

        grids = self.get_init_grid(batch_size=self.batch_size,\
                 dim = target.shape[-2])

        my_training = {}
        my_training["lr"] = lr
        my_training["max_ca_steps"] = max_ca_steps
        my_training["batch_size"] = batch_size
        my_training["number_channels"] = self.number_channels
        my_training["number_hidden_channels"] = self.number_hidden

        my_training["step"] = []
        my_training["loss"] = []
    
        for step in range(max_steps):

            with torch.no_grad():
                batch_index = np.random.choice(len(grids), 4, replace=False)

                x = grids

                if step % 8 == 0:
                    x[:1] = self.get_init_grid(batch_size=1, dim=x.shape[-2])


            self.optimizer.zero_grad()

            for ca_step in range(np.random.randint(1,16) + max_ca_steps):
                x = self.forward(x)

            grams_pred = compute_grams(x, device=self.my_device)
            grams_target = compute_grams(target, device=self.my_device)

            loss = compute_style_loss(grams_pred, grams_target)
            loss.backward()

            for p in self.parameters():
                p.grad /= (p.grad.norm()+1e-8)


            self.optimizer.step()
            self.lr_scheduler.step()

            if step % display_every == 0:
                print(f"loss at step {step} = {loss:.4e}")
            
            my_training["loss"].append(loss)
            my_training["step"].append(step)

            _ = self.save_log(my_training, exp_tag)

    def count_parameters(self):

        number_parameters = 0

        for param in self.parameters():
            number_parameters += param.numel()

        return number_parameters


    def to_device(self, my_device):

        
        if "cuda" in torch.device(my_device).type \
                and torch.cuda.is_available():
            self.my_device= torch.device(my_device)
        elif "cuda" in torch.device(my_device).type:
            print(f"warning, cuda not found but {my_device} specified, falling back to cpu")
            self.my_device = torch.device("cpu")
        else:
            self.my_device = torch.device(my_device)

        self.to(self.my_device)
        self.filters = self.filters.to(self.my_device)

    def save_parameters(self, save_path):

        torch.save(self.state_dict(), save_path)

    def load_parameters(self, load_path):

        state_dict = torch.load(load_path)

        self.load_state_dict(state_dict)
        
if __name__ == "__main__": #pragma: no cover

    nca = NCA()

    grid = nca.get_init_grid()

    for step in range(10):
        grid = nca(grid)

    print("OK")

