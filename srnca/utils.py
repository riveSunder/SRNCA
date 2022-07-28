import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import skimage
import skimage.io as sio
import skimage.transform

vgg16 = torchvision.models.vgg16(pretrained=True).features

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

def seed_all(my_seed=42):
    
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)

def compute_grams(imgs, device="cpu"):
    
    style_layers = [1, 6, 11, 18, 25]  
    vgg16.to(torch.device(device))
    

    img_mean = (1e-9 + imgs).mean(dim=(0,2,3))[None,:,None, None]

    x = imgs
    
    grams = []

    if x.shape[1] >= 3: 
        # random matrix adapter for single or multichannel tensors
        #restore_seed = torch.seed()
        #orch.manual_seed(42)
        #w_adapter = torch.rand( 3, x.shape[1], 1,1, device=torch.device(device)) #3, 3)

        #x = F.conv2d(x, w_adapter) 

        #torch.manual_seed(restore_seed)
        x = x[:,:3,:,:]

    elif x.shape[1] < 3: 
        while x.shape[1] < 3:
            x = torch.cat([x,x], dim=1)
        x = x[:,:3,:,:]

    for i, layer in enumerate(vgg16[:max(style_layers)+1]):

        x = layer(x)
        if i in style_layers:
            
            h, w = x.shape[-2:]
            y = x.clone()  # workaround for pytorch in-place modification bug(?)
            
            gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h*w)
            grams.append(gram)
            
    return grams

def compute_style_loss(grams_pred, grams_target):
    
    loss = 0.0
    
    for x, y in zip(grams_pred, grams_target):
        loss = loss + (x-y).square().mean()
        
    return loss


def read_image(url, max_size=None):
    
    img = sio.imread(url)
    if len(img.shape) == 2:
        img = img[:,:,None]
    
    dim_x, dim_y = img.shape[0], img.shape[1]

    min_dim = min([dim_x, dim_y])
    img = img[:min_dim, :min_dim]

    if max_size is not None:
        img = skimage.transform.resize(img, (max_size, max_size))
   
    img = np.float32(img)/ img.max()
    
    return img

def image_to_tensor(img, device="cpu"):

    if len(img.shape) == 2:
        my_tensor = torch.tensor(img[np.newaxis, np.newaxis, ...], \
                device=torch.device(device))
    elif len(img.shape) == 3:
        my_tensor = torch.tensor(img.transpose(2,0,1)[np.newaxis,...], \
                device=torch.device(device))
    
    return my_tensor

def tensor_to_image(my_tensor, index=0):

    if my_tensor.shape[1] == 1:
        # monochrome images
        img = my_tensor[index,0,:,:].detach().cpu().numpy()
    else:
        # rgb or rgba images, convert to rgb
        img = my_tensor[index,:3,:,:].permute(1,2,0).detach().cpu().numpy()

    return img

def perceive(x, filters):
    
    batch, channels, height, width = x.shape
    
    x = x.reshape(batch*channels, 1, height, width)
    x = F.pad(x, (1,1,1,1), mode="circular")
    
    x = F.conv2d(x, filters[:, np.newaxis, :, :])
    
    perception = x.reshape(batch, -1, height, width)
    
    return perception

