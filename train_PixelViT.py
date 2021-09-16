#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#
import matplotlib.pyplot as plt
import argparse
import math
import sys
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from image_datasets_PixelViT import add_dataset_arguments, get_mnist

from utils import get_optimizer, EpochStats, save_model, discretized_mix_logistic_loss as dmll

from utils import sample_mol

def loss(y, y_hat):
    log2 = 0.6931471805599453
    #print(len(y_hat))
    y_hat = y_hat.permute(0, 2, 1).contiguous()
    N, C, L = y_hat.shape
    l = dmll(y_hat, y.view(N, L, 1))
    bpd = l.item() / log2
    return l, bpd

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_bpd = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            l, bpd = loss(y, y_hat)
            total_loss += x.shape[0] * l.item()
            total_bpd += x.shape[0] * bpd
            total_samples += x.shape[0]
    if sys.stdout.isatty():
        print()
    print(
        "Testing =>",
        "Loss:",
        total_loss/total_samples,
        "bpd:",
        total_bpd/total_samples
    )

    return total_loss/total_samples


def train(model, optimizer, dataloader, iteration, total_iterations, device):
    model.train()
    all_losses = []
    for epoch in range(args.n_epochs):
        
        losses = 0
        with tqdm(total=len(dataloader)) as pbar:
            for i, (x, y) in enumerate(dataloader):    
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                #print(x.size())

                y_hat = model(x)
                
                #print(y.size(), y_hat.size())
                
                l, bpd = loss(y, y_hat)
                l.backward()
                optimizer.step()        
                losses += l.item()
                pbar.set_description('Loss: {}'.format(l.item()))
                pbar.update(1)
        
                if i % 64 == 0:
                    #print(y_hat.size())
                    img = sample_mol(y_hat[0])
                    img = img.detach().cpu().numpy().squeeze()
                    img = img.reshape((32,32))
                    plt.subplot(1,3,1)
                    plt.imshow(img)
                    
                    plt.subplot(1,3,2)
                    img = y[0].detach().cpu().numpy().squeeze()
                    img = img.reshape((32, 32))
                    plt.imshow(img)
                    
                    plt.subplot(1,3,3)
                    img = x[0].detach().cpu().numpy().squeeze()
                    img = img.reshape((32, 32))
                    plt.imshow(img)
                    plt.show()          
        
        
        avg_loss = losses/len(dataloader)
        #print('-')
        #print(epoch, '- Loss: {}'.format(avg_loss))
        all_losses.append(avg_loss)
        save_model(args.save_to, model, optimizer, epoch)
        plt.plot(all_losses)
        plt.show()
        print('Loss:', avg_loss)

def main(args):
    import os
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    #print('Visible Device:', os.environ['CUDA_VISIBLE_DEVICES'])

    # Make the dataset and the model
    train_loader, test_loader = get_mnist()
    
    ################## MODEL HERE ##################
    '''
    from gpt2_v1 import GPT2
    model = GPT2(embed_dim = 16,
                 num_heads = 4,
                 num_layers = 8,
                 num_vocab = 256,
                 num_classes = 30,
                 num_positions = (28*28) - 1)
    '''
    '''
    from gpt2_v2 import Transformer
    model = Transformer( dims = 64,            # embed_dim
                         heads = 4,            # num_heads
                         layers = 8,           # num_layers
                         words = 256,          # num_vocab
                         seq_len = 783,        # num_positions
                         bidirectional = False,
                         dropout = 0.1,
                         rate   = 4,
                         pad_idx = 1,
                         num_classes = 30)
    '''
    
    #from gpt2_v3 import GPT2Model
    #model = GPT2Model(args)
    
    #from linear_v1 import Linear
    #model = Linear()
    
    from PixelViT2 import Transformer
    
    model = Transformer( channels = 1,
                         patch_size = 4,
                         image_size = 32,
                         dim = 1024,
                         heads = 8,            # num_heads
                         layers = 8,           # num_layers
                         words = 256,           # num_vocab
                         seq_len = 8*8,      # num_positions
                         bidirectional = False,
                         dropout = 0.1,
                         rate   = 4,
                         pad_idx = 1,
                         num_classes = 30)
    
    # Choose a device and move everything there

    print("Running on {}".format(args.device))
    model.to(args.device)

    optimizer = get_optimizer(model.parameters(), args)
    iteration = 0

    #optimizer.set_lr(args.lr)

    yielded = train(
        model,
        optimizer,
        train_loader,
        iteration,
        args.iterations,
        args.device
    )

    # Non-zero exit code to notify the process watcher that we yielded
    if yielded:
        sys.exit(1)


if __name__ == "__main__":
    from parameters import Parameters
    args = Parameters()
    main(args)


