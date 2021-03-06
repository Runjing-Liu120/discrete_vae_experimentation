import numpy as np
import scipy.stats as stats
import os

import torch
print('torch version', torch.__version__)

import time
import json

from torch.utils.data import Dataset, DataLoader, sampler

import mnist_data_lib
import mnist_vae_lib
import common_utils
import se
import distutils.util
import argparse


parser = argparse.ArgumentParser(description='FullVAE')

parser.add_argument('--mnist_data_dir', type = str,
                    default='../mnist_data/')
parser.add_argument('--latent_dim', type=int, default=5, metavar='N',
                    help='laten dimension (default = 5)')
parser.add_argument('--set_true_class_label', type=distutils.util.strtobool, default='False',
                    help='whether to set the class label ')


# Training parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-6)
parser.add_argument('--dropout', type = float, default = 0.5)
parser.add_argument('--learning_rate', type = float, default = 0.001)

# saving encoder
parser.add_argument('--outdir', type = str,
                    default='./', help = 'directory for saving encoder and decoder')
parser.add_argument('--outfilename', type = str,
                    default='enc',
                    help = 'filename for saving the encoder and decoder')
parser.add_argument('--save_every', type = int, default = 50,
                    help='save encoder ever how ___ epochs (default = 50)')

# Loading encoder
parser.add_argument('--load_enc', type=distutils.util.strtobool, default='False',
                    help='whether to load encoder')
parser.add_argument('--enc_init', type = str,
                    help = 'file from which to load encoder')
parser.add_argument('--load_dec', type=distutils.util.strtobool, default='False',
                    help='whether to load decoder')
parser.add_argument('--dec_init', type = str,
                    help = 'file from which to load encoder')

# Whether to just work with subset of data
parser.add_argument('--propn_sample', type = float,
                    help='proportion of dataset to use',
                    default = 1.0)

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outdir)

    if args.load_enc:
        assert os.path.isfile(args.enc_init)

    if args.load_dec:
        assert os.path.isfile(args.dec_init)

validate_args()

np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

# LOAD DATA
print('Loading data')
# train_set, test_set = mnist_data_lib.load_mnist_data(data_dir = args.mnist_data_dir)
train_set, test_set = mnist_data_lib.get_mnist_dataset(data_dir = args.mnist_data_dir,
                                                        propn_sample=args.propn_sample)

batch_size = args.batch_size
train_loader = DataLoader(
                 dataset=train_set,
                 batch_size=args.batch_size,
                 shuffle=True)

test_loader = DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

# SET UP VAE
slen = train_set[0]['image'].shape[-1]
latent_dim = args.latent_dim
n_classes = 10
vae = mnist_vae_lib.HandwritingVAE(latent_dim = latent_dim,
                            n_classes = n_classes,
                            slen = slen)
vae.cuda()
if args.load_enc:
    print('initializing encoder from ', args.enc_init)

    vae.encoder.load_state_dict(torch.load(args.enc_init,
                                    map_location=lambda storage, loc: storage))
if args.load_dec:
    print('initializing decoder from ', args.dec_init)

    vae.decoder.load_state_dict(torch.load(args.dec_init,
                                    map_location=lambda storage, loc: storage))

print('training vae')
print('set_true_class_label: ', args.set_true_class_label)

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)
vae.train_module(train_loader, test_loader,
                    n_epoch = args.epochs,
                    print_every = 10,
                    outfile = outfile,
                    save_every = args.save_every,
                    weight_decay = args.weight_decay,
                    lr = args.learning_rate,
                    set_true_class_label = args.set_true_class_label)

print('done. Total time: {}secs'.format(time.time() - t0_train))
