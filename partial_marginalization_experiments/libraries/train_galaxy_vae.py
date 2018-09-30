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
import celeste_net

from datasets import Synthetic

import galaxy_experiments_lib as galaxy_lib

import os

parser = argparse.ArgumentParser(description='CelesteNet')

# training options
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
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

# if warm start_time
parser.add_argument('--galaxy_enc_warm_start', type = distutils.util.strtobool,
                    default = False, help = 'whether to initialize the galaxy encoder')
parser.add_argument('--galaxy_enc_init', type = str,
                    default = '../galaxy_warm_starts/galaxy_enc_init.dat',
                    help = 'file from which to load galaxy encoder')

parser.add_argument('--galaxy_dec_warm_start', type = distutils.util.strtobool,
                    default = False, help = 'whether to initialize the galaxy decoder')
parser.add_argument('--galaxy_dec_init', type = str,
                    default = '../galaxy_warm_starts/galaxy_dec_init.dat',
                    help = 'file from which to load galaxy decoder')

parser.add_argument('--attn_enc_warm_start', type = distutils.util.strtobool,
                    default = False, help = 'whether to initialize the attention encoder')
parser.add_argument('--attn_enc_init', type = str,
                    default = '../galaxy_warm_starts/attn_enc_init.dat',
                    help = 'file from which to load galaxy encoder')


# seed
parser.add_argument('--seed', type=int, default=64, metavar='S',
                    help='random seed (default: 64)')

args = parser.parse_args()


_ = torch.manual_seed(args.seed)
np.random.seed(args.seed)

def validate_args():
    assert os.path.exists(args.vae_outdir)

    if args.attn_enc_warm_start:
        assert os.path.isfile(args.attn_enc_init)

    if args.galaxy_enc_warm_start:
        assert os.path.isfile(args.galaxy_enc_init)

    if args.galaxy_dec_warm_start:
        assert os.path.isfile(args.galaxy_dec_init)


validate_args()

# get dataset
ds = Synthetic(args.slen, min_galaxies=1, max_galaxies=1, mean_galaxies=1, num_images=12800)
train_loader, test_loader = galaxy_lib.get_train_test_data(ds, batch_size=args.batchsize)

# set up vae
galaxy_vae = celeste_net.OneGalaxyVAE(args.slen)

# if warm start
if args.attn_enc_warm_start:
    print('loading attention encoder from ' + args.attn_enc_init)
    state_dict = torch.load(args.attn_enc_init, map_location='cpu')
    galaxy_vae.attn_enc.load_state_dict(state_dict, strict=False)

if args.galaxy_enc_warm_start:
    print('loading galaxy encoder from ' + args.galaxy_enc_init)
    state_dict = torch.load(args.galaxy_enc_init, map_location='cpu')
    galaxy_vae.enc.load_state_dict(state_dict, strict=False)

if args.galaxy_dec_warm_start:
    print('loading galaxy decoder from ' + args.galaxy_dec_init)
    state_dict = torch.load(args.galaxy_dec_init, map_location='cpu')
    galaxy_vae.dec.load_state_dict(state_dict, strict=False)



galaxy_rnn = galaxy_lib.CelesteRNN(args.slen, one_galaxy_vae=galaxy_vae)
galaxy_rnn.cuda()

print("training the one-galaxy autoencoder...")
filename = args.vae_outdir + args.vae_outfilename
galaxy_lib.train_module(galaxy_rnn, train_loader, test_loader,
                        epochs = args.epochs,
                        save_every = args.save_every,
                        alpha = 0.0,
                        topk = args.topk,
                        use_baseline = True,
                        lr = 1e-2,
                        weight_decay = 1e-6,
                        filename = filename,
                        seed = args.seed)
