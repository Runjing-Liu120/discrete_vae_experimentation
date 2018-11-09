import numpy as np
import scipy.stats as stats
import os

import torch
print('torch version', torch.__version__)

import time
import json

from torch.utils.data import Dataset, DataLoader, sampler

import cifar_data_utils
import cifar_semisupervised_lib


import common_utils
import semisupervised_vae_lib as ss_vae_lib

import distutils.util
import argparse


parser = argparse.ArgumentParser(description='FullVAE')

parser.add_argument('--cifar_data_dir', type = str,
                    default='../cifar100_data/')

# Training parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-6)
parser.add_argument('--learning_rate', type = float, default = 0.001)

parser.add_argument('--propn_labeled', type = float, default = 0.1,
                    help = 'proportion of training data labeled')

parser.add_argument('--alpha', type = float, default = 1.0,
                    help = 'weight of cross_entropy_term')

parser.add_argument('--topk', type=int, default=0,
                    help='number of weights to sample for reinforce')
parser.add_argument('--use_baseline', type=distutils.util.strtobool, default='True',
                    help='whether or not to add a use_baseline')

parser.add_argument('--use_true_labels', type=distutils.util.strtobool, default='False',
                    help='for debugging only: whether to give the procedure all the labels')

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
    cifar_data_utils.load_semisupervised_cifar_dataset(propn_sample = args.propn_sample,
                                                    propn_labeled = args.propn_labeled)

if args.propn_labeled == 1:
    print('all images are labeled. ')
    train_loader_unlabeled = None
    labeled_batchsize = args.batch_size
else:
    train_loader_unlabeled = torch.utils.data.DataLoader(
                     dataset=train_set_unlabeled,
                     batch_size=args.batch_size,
                     shuffle=True)
    print('num_train_unlabeled: \n', train_set_unlabeled.num_images)

    labeled_batchsize = round(args.propn_labeled / (1 - args.propn_labeled))

    print('len(train_loader_unlabeled): ', len(train_loader_unlabeled))


train_loader_labeled = torch.utils.data.DataLoader(
                 dataset=train_set_labeled,
                 batch_size=labeled_batchsize,
                 shuffle=True)
print('len(train_loader_labeled): ', len(train_loader_labeled))

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

print('num_train_labeled: ', train_set_labeled.num_images)
# print('check: \n', data_labeled['image'].shape[0])

print('num_test: ', test_set.num_images)

# SET UP VAE
print('setting up VAE: ')
image_config = {'slen': 32,
                 'channel_num': 3,
                 'n_classes': 100}

cond_vae_config = {'kernel_num': 128,
                   'z_size': 128}

# classifier_config = {'depth': 28,
#                      'widen_factor': 1,
#                      'dropout_rate': 0.3}
classifier_config = {'depth': 100,
                     'k': 12}

print('image_config', image_config)
print('cond_vae_config', cond_vae_config)
print('classifier_config', classifier_config)

vae = \
    cifar_semisupervised_lib.get_cifar_semisuperivsed_vae(image_config,
                                                            cond_vae_config,
                                                            classifier_config)

vae.to(device)
if args.load_enc:
    print('initializing encoder from ', args.enc_init)

    vae.conditional_vae.encoder.load_state_dict(torch.load(args.enc_init,
                                    map_location=lambda storage, loc: storage))
if args.load_dec:
    print('initializing decoder from ', args.dec_init)

    vae.conditional_vae.decoder.load_state_dict(torch.load(args.dec_init,
                                    map_location=lambda storage, loc: storage))
if args.load_classifier:
    print('initializing classifier from ', args.classifier_init)

    vae.classifier.load_state_dict(torch.load(args.classifier_init,
                                    map_location=lambda storage, loc: storage))


print('training vae')

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)

ss_vae_lib.train_semisupervised_model(vae,
                    train_loader_unlabeled = train_loader_unlabeled,
                    train_loader_labeled = train_loader_labeled,
                    test_loader = test_loader,
                    n_epoch = args.epochs,
                    alpha = args.alpha,
                    outfile = outfile,
                    save_every = args.save_every,
                    weight_decay = args.weight_decay,
                    lr = args.learning_rate,
                    topk = args.topk,
                    use_baseline = args.use_baseline,
                    train_classifier_only = args.train_classifier_only,
                    use_true_labels = args.use_true_labels)

print('done. Total time: {}secs'.format(time.time() - t0_train))
