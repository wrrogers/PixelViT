import matplotlib.pyplot as plt
import os
import cv2

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from PIL import Image

from PixelViT import Transformer
from torch.utils.data import DataLoader
from utils import sample_mol
from image_datasets_PixelViT import get_mnist

from parameters import Parameters

import cv2

def get_patch(img, patch_size, patch_x, patch_y):
    x = patch_x * patch_size
    y = patch_y * patch_size
    patch = img[x:x+patch_size, y:y+patch_size]
    return patch

def apply_patch(img, patch_size, patch_x, patch_y, patch):
    #print('---------------------')
    #print(patch_x)
    #print(patch_y)
    x = (patch_x) * patch_size
    if patch_y < 7:
        y = (patch_y+1) * patch_size
    else:
        if patch_x < 7:
            y = 0
            x = (patch_x+1) * patch_size
        else:
            return img
    #print('x:', x)
    #print('y:', y)
    img[:, :, x:x+patch_size, y:y+patch_size] = patch
    #print('---------------------')
    return img

def to_negpos(img):
    img = img/255
    img = img*2
    img = img-1
    return img

def sample(model, img, n_patches):
    plt.imshow(img[0].detach().cpu().numpy().squeeze())
    plt.show()
    targ = img[0].clone()
    targ = targ.unsqueeze(0)
    targ[:, :, 8:] = -1
    plt.imshow(targ[0].detach().cpu().numpy().squeeze())
    plt.show()
    with torch.no_grad():
        for n in range(n_patches//4, n_patches):
            for m in range(0, n_patches):
                #print(n, m)
                y_hat = model(targ)
                #print('-')
                #print('Y_hat:', y_hat.size())
                out = sample_mol(y_hat[0]).squeeze()
                #print('OUT:', out.size())
                out = out.reshape((32, 32))
                patch = get_patch(out, 4, n, m)
                #print('patch', patch.size())

                #plt.title('patch')
                #plt.imshow(patch.detach().cpu().numpy().squeeze())
                #plt.show()
                #print(patch)
                patch = to_negpos(patch)

                targ = apply_patch(targ, 4, n, m, patch)
                
                plt.subplot(1,3,1)
                plt.imshow(out.detach().cpu().numpy().squeeze())
                plt.subplot(1,3,2)
                plt.imshow(targ.detach().cpu().numpy().squeeze())
                plt.subplot(1,3,3)
                plt.imshow(img[0].detach().cpu().numpy().squeeze())
                plt.show()

                
                #print('OUT:', out.size())
                
    return targ

def make_figure(rows, centroids):
    figure = np.stack(rows, axis=0)
    print(figure.shape)
    rows, cols, h, w = figure.shape
    #figure = unquantize(figure.swapaxes(1, 2).reshape(h * rows, w * cols), centroids)
    figure = (figure * 256).astype(np.uint8)
    return Image.fromarray(np.squeeze(figure))

#def main(args):

out_dir = r'C:\Users\william\ImageGPT\imgs'

args = Parameters()
model = Transformer( channels = 1,
                     patch_size = 4,
                     image_size = 32,
                     dim = 1024,
                     heads = 8,            # num_heads
                     layers = 16,           # num_layers
                     words = 256,           # num_vocab
                     seq_len = 8*8,      # num_positions
                     bidirectional = False,
                     dropout = 0.1,
                     rate   = 4,
                     pad_idx = 1,
                     num_classes = 30)
    
checkpoint = torch.load(r'models\model_H8L16D1024-155.pth')
#print(checkpoint.keys())
model.load_state_dict(checkpoint['model_state'])

#train_dl, valid_dl, test_dl = get_dataloaders(1)

#dl = iter(valid_dl)

# rows for figure
rows = []

for example in tqdm(range(1)):

    train_loader, test_loader = get_mnist()
    
    data, targ = next(iter(train_loader))

    print('data:', data.size())

    img = data[13:14]
    
    out = model(img)
    out = sample_mol(out[0]).squeeze()

    plt.subplot(1,2,1)
    plt.imshow(out.detach().cpu().numpy().squeeze().reshape((32, 32)))
    plt.subplot(1,2,2)
    plt.imshow(img.detach().cpu().numpy().squeeze())
    plt.show()
    
    print(img.shape, img.min(), img.max())
    print(img.dtype)

    # first half of image is context
    #img[:, :, half:, :] = 0

    #savimg = (img.detach().cpu().numpy().squeeze() * 255).astype(np.uint8)
    #cv2.imwrite(path, savimg)
    
    # predict second half of image
    print(img.size())
    preds = sample(model, img, n_patches=8)

    plt.imshow(preds[0].detach().cpu().numpy().squeeze())
    plt.show()
        # combine context, preds, and truth for figure
        #rows.append(
        #    np.concatenate([context[None, ...], preds, img[None, ...]], axis=0)
        #)
    
    
    #figure = make_figure(rows, centroids)
    #figure.save("figure.png")


#if __name__ == "__main__":
    
#    main(args)
