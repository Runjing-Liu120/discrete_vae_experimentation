#!/usr/bin/env python3

import argparse
import numpy as np
import timeit
import pathlib

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim

import sys
sys.path.insert(0, '../../../celeste_net/')

from datasets import Synthetic

import galaxy_experiments_lib as galaxy_lib


parser = argparse.ArgumentParser(description='CelesteNet')

# training options
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--topk', type = int, default = 5,
                    help='how many to integrate out')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')

# data parameters
parser.add_argument('--slen', type=int, default=31, metavar='N',
                    help='image height and width (default 31)')

# saving options
parser.add_argument('--vae_outdir', type = str,
                    default='./', help = 'directory for saving vae')
parser.add_argument('--vae_outfilename', type = str,
                    default='galaxy_vae',
                    help = 'filename for saving the vae')
parser.add_argument('--save_every', type = int, default = 50,
                    help='save encoder ever how ___ epochs (default = 50)')

# seed
parser.add_argument('--seed', type=int, default=64, metavar='S',
                    help='random seed (default: 64)')

args = parser.parse_args()


_ = torch.manual_seed(args.seed)
np.random.seed(args.seed)

def validate_args():
    assert os.path.exists(args.vae_outdir)

validate_args()

# get dataset
ds = Synthetic(args.slen, min_galaxies=1, max_galaxies=1, mean_galaxies=1, num_images=12800)
train_loader, test_loader = galaxy_lib.get_train_test_data(ds, batch_size=args.batch_size)

vae = galaxy_lib.CelesteRNN(args.slen, max_detections=4)

print("training the one-galaxy autoencoder...")
galaxy_lib.train_module(vae, train_loader, test_loader,
                        epochs = args.epochs,
                        save_every = args.save_every,
                        alpha = 0.0,
                        topk = args.topk,
                        use_baseline = True,
                        lr = 1e-4,
                        weight_decay = 1e-6,
                        filename = './test',
                        seed = args.seed)
