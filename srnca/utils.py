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

def compute_grams(imgs):
    
    style_layers = [1, 6, 11, 18, 25]  
    
    # from https://github.com/google-research/self-organising-systems
    # no idea why
    # -> removed, left as comment as a reminder to look into it.
    #mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    #std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]

    img_mean = (1e-9 + imgs).mean(dim=(0,2,3))[None,:,None, None]

    #x = (imgs-img_mean) / imgs.std()
    x = (imgs - 0.445) / 0.226
    
    grams = []

    if x.shape[1] != 3: 
        # random matrix adapter for single or multichannel tensors
        restore_seed = torch.seed()
        torch.manual_seed(42)
        w_adapter = torch.rand( 3, x.shape[1], 1,1) #3, 3)

        x = F.conv2d(x, w_adapter) #, padding=1, padding_mode="circular")

        torch.manual_seed(restore_seed)

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
    
    if max_size is not None:
        img = skimage.transform.resize(img, (max_size, max_size))
    
   
    img = np.float32(img)/ img.max()
    
    return img

def image_to_tensor(img):

    if len(img.shape) == 2:
        my_tensor = torch.tensor(img[np.newaxis, np.newaxis, ...])
    elif len(img.shape) == 3:
        my_tensor = torch.tensor(img.transpose(2,0,1)[np.newaxis,...])
    
    return my_tensor

def tensor_to_image(my_tensor, index=0):

    if my_tensor.shape[1] == 1:
        # rgb or rgba images, convert to rgb
        img = my_tensor[index,0,:,:]
    else:
        # monochrome images
        img = my_tensor[index,:3,:,:].permute(1,2,0).detach().numpy()

    return img

def perceive(x, filters):
    
    batch, channels, height, width = x.shape
    
    x = x.reshape(batch*channels, 1, height, width)
    x = F.pad(x, (1,1,1,1), mode="circular")
    
    x = F.conv2d(x, filters[:, np.newaxis, :, :])
    
    perception = x.reshape(batch, -1, height, width)
    
    return perception

