import numpy as np
import scipy.stats as stats
import os

import torch
print('torch version', torch.__version__)

import time
import timeit

import json

from torch.utils.data import Dataset, DataLoader, sampler

import cifar_data_utils
import cifar_semisupervised_lib

import mnist_utils

import common_utils
import semisupervised_vae_lib as ss_vae_lib

import distutils.util
import argparse


parser = argparse.ArgumentParser(description='FullVAE')

parser.add_argument('--use_cifar100', type=distutils.util.strtobool, default='True')

# parser.add_argument('--cifar_data_dir', type = str,
#                     default='../cifar100_data/')

# Training parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-6)
parser.add_argument('--learning_rate', type = float, default = 0.001)


# saving encoder
parser.add_argument('--outdir', type = str,
                    default='./', help = 'directory for saving encoder and decoder')
parser.add_argument('--outfilename', type = str,
                    default='enc',
                    help = 'filename for saving the encoder and decoder')
parser.add_argument('--save_every', type = int, default = 50,
                    help='save encoder ever how ___ epochs (default = 50)')


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


validate_args()

np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

# LOAD DATA
if args.use_cifar100:
    print('Loading cifar 100')
    train_set, _, test_set = \
        cifar_data_utils.load_semisupervised_cifar_dataset(
                                        cifar100 = True,
                                        data_dir = '../cifar100_data',
                                        propn_sample = args.propn_sample,
                                        propn_labeled = 1.0)

    n_classes = 100
else:
    print('Loading cifar 10')
    train_set, _, test_set = \
        cifar_data_utils.load_semisupervised_cifar_dataset(
                                        cifar100 = False,
                                        data_dir = '../cifar10_data',
                                        propn_sample = args.propn_sample,
                                        propn_labeled = 1.0)

    n_classes = 10


train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=args.batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

print('num_train_labeled: ', train_set.num_images)
print('num_test: ', test_set.num_images)

# SET UP VAE
print('setting up VAE: ')
image_config = {'use_cifar100', args.use_cifar100,
                'slen': 32,
                 'channel_num': 3,
                 'n_classes': n_classes}

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

print('training vae')

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)


for epoch in range(1, n_epoch + 1):
    optimizer = optim.Adam([
            {'params': vae.conditional_vae.parameters(), 'lr': args.learning_rate}],
            weight_decay=args.weight_decay)

    for batch_idx, data in enumerate(train_loader):
        images = data['image'].to(device)
        labels = data['label'].to(device)

        # forward
        vae.train()
        latent_means, latent_std, latent_samples, image_mean, image_var = \
            vae.cond_vae.forward(images, labels)

        # get loss
        image_bern = image * cifar_data_utils.CIFAR10_STD_TENSOR.to(device) + \
                        cifar_data_utils.CIFAR10_mean_TENSOR.to(device)
        assert torch.min(image_bern) > -1e-5

        recon_loss = -mnist_utils.get_bernoulli_loglik(pi = image_mean,
                                                        x = image_bern)

        kl_term = mnist_utils.get_kl_q_standard_normal(latent_means, \
                                                            latent_std)
        loss = recon_loss + kl_term

        loss.backward()

        avg_loss += loss / train_set.num_images

    print('epoch: {}, loss: {}'.format(epoch, avg_loss))

    if epoch % 10 == 0:
        torch.save(vae.conditional_vae.state_dict(), outfile)
        with open('../cifar10_data/test_batch_cifar10.pkl', 'rb') as f:
            data_labeld_r = pickle.load(f)
        vae.eval()
        print('debugging loss: ', vae.get_conditional_loss(data_labeld_r['image'].to(device),
                                    data_labeld_r['label'].to(device)).mean())

print('done. Total time: {}secs'.format(time.time() - t0_train))
