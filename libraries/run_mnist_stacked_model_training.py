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

import distutils.util
import argparse

import torch.optim as optim

parser = argparse.ArgumentParser(description='FullVAE')

parser.add_argument('--mnist_data_dir', type = str,
                    default='../mnist_data/')
parser.add_argument('--sqrt_latent_dim1', type=int, default=5, metavar='N',
                    help='latent dimension (default = 5)')
parser.add_argument('--latent_dim2', type=int, default=5, metavar='N',
                    help='latent dimension (default = 5)')

parser.add_argument('--model1_enc_init', type=str,
                    help='where to load the model 1 vae')
parser.add_argument('--model1_dec_init', type=str,
                    help='where to load the model 1 vae')

# Training parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-6)
parser.add_argument('--dropout', type = float, default = 0.5)
parser.add_argument('--learning_rate', type = float, default = 0.001)

parser.add_argument('--propn_labeled', type = float, default = 0.1,
                    help = 'proportion of training data labeled')
parser.add_argument('--alpha', type = float, default = 1.0,
                    help = 'weight of cross_entropy_term')

parser.add_argument('--num_reinforced', type=int, default=0,
                    help='number of weights to sample for reinforce')
parser.add_argument('--use_baseline', type=distutils.util.strtobool, default='False',
                    help='whether or not to add a use_baseline')

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
parser.add_argument('--load_classifier', type=distutils.util.strtobool, default='False',
                    help='whether to load classifier')
parser.add_argument('--classifier_init', type = str,
                    help = 'file from which to load encoder')
parser.add_argument('--train_classifier_only', type=distutils.util.strtobool, default='False',
                    help = 'whether to train classifier')

# Whether to just work with subset of data
parser.add_argument('--propn_sample', type = float,
                    help='proportion of dataset to use',
                    default = 1.0)

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outdir)

    assert os.path.isfile(args.model1_enc_init)
    assert os.path.isfile(args.model1_dec_init)

    if args.load_enc:
        assert os.path.isfile(args.enc_init)

    if args.load_dec:
        assert os.path.isfile(args.dec_init)

    if args.load_classifier:
        assert os.path.isfile(args.classifier_init)

validate_args()

np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

# LOAD DATA
print('Loading data')
train_set_labeled, train_set_unlabeled, test_set = \
    mnist_data_lib.get_mnist_dataset_semisupervised(propn_sample = args.propn_sample,
                                                    propn_labeled = args.propn_labeled)

train_loader_labeled = torch.utils.data.DataLoader(
                 dataset=train_set_labeled,
                 batch_size=len(train_set_labeled),
                 shuffle=True)

train_loader_unlabeled = torch.utils.data.DataLoader(
                 dataset=train_set_unlabeled,
                 batch_size=args.batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

for batch_idx, d in enumerate(train_loader_labeled):
    data_labeled = d
    break

print('num_train_labeled: ', train_set_labeled.num_images)
print('num_train_unlabeled: \n', train_set_unlabeled.num_images)

print('num_test: ', test_set.num_images)

# SET UP VAE
model1_vae = stacked_vae_lib.Model1VAE(sqrt_latent_dim = args.sqrt_latent_dim1)
print('initializing model1 enc from ', args.model1_enc_init)
print('initializing model2 dec from ', args.model1_dec_init )

enc_file = args.model1_enc_init
dec_file = args.model1_dec_init

model1_vae.encoder.load_state_dict(torch.load(enc_file,
                               map_location=lambda storage, loc: storage))

model1_vae.decoder.load_state_dict(torch.load(dec_file,
                               map_location=lambda storage, loc: storage))


slen = train_set_unlabeled[0]['image'].shape[0]
latent_dim2 = args.latent_dim2
n_classes = 10
vae = stacked_vae_lib.StackedModelVAE(model1_vae,
                m2_latent_dim = latent_dim2,
                n_classes = n_classes,
                use_baseline = args.use_baseline)

vae.to(device)

# Set up optimizer
if args.train_classifier_only:
    # for debugging only
    optimizer = optim.Adam([
            {'params': vae.classifier.parameters(), 'lr': args.learning_rate}],
            weight_decay=args.weight_decay)

    if vae.use_baseline:
        optimizer = optim.Adam([
                {'params': vae.classifier.parameters(), 'lr': args.learning_rate},
                {'params': vae.baseline_learner.parameters(), 'lr': args.learning_rate}],
                weight_decay=args.weight_decay)

else:
    optimizer = optim.Adam([
            {'params': vae.classifier.parameters(), 'lr': args.learning_rate},
            {'params': vae.encoder.parameters(), 'lr': args.learning_rate},
            {'params': vae.decoder.parameters(), 'lr': args.learning_rate}],
            weight_decay=args.weight_decay)

    if vae.use_baseline:
        optimizer = optim.Adam([
                {'params': vae.classifier.parameters(), 'lr': args.learning_rate},
                {'params': vae.encoder.parameters(), 'lr': args.learning_rate},
                {'params': vae.decoder.parameters(), 'lr': args.learning_rate},
                {'params': vae.baseline_learner.parameters(), 'lr': args.learning_rate}],
                weight_decay=args.weight_decay)



print('training vae')

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)
mnist_vae_lib.train_semisupervised_model(vae, optimizer
                    train_loader_unlabeled = train_loader_unlabeled,
                    test_loader = test_loader,
                    labeled_images = data_labeled['image'],
                    labels = data_labeled['label'],
                    n_epoch = args.epochs,
                    alpha = args.alpha,
                    outfile = outfile,
                    save_every = args.save_every,
                    num_reinforced = args.num_reinforced)

print('done. Total time: {}secs'.format(time.time() - t0_train))
