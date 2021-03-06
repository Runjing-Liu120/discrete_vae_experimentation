import numpy as np
import scipy.stats as stats
import os

import torch
print('torch version', torch.__version__)

import torch.optim as optim

import time
import json

from torch.utils.data import Dataset, DataLoader, sampler

import mnist_data_lib
import mnist_vae_lib
import common_utils

import distutils.util
import argparse


parser = argparse.ArgumentParser(description='FullVAE')

parser.add_argument('--mnist_data_dir', type = str,
                    default='../mnist_data/')

# Training parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')

parser.add_argument('--weight_decay', type = float, default = 1e-6)
parser.add_argument('--learning_rate', type = float, default = 0.001)

# saving encoder
parser.add_argument('--outdir', type = str,
                    default='./', help = 'directory for saving encoder and decoder')
parser.add_argument('--outfilename', type = str,
                    default='enc',
                    help = 'filename for saving the encoder and decoder')

# inits
parser.add_argument('--load_classifier', type=distutils.util.strtobool, default='False',
                    help='whether to load classifier')
parser.add_argument('--classifier_init', type = str,
                    help = 'file from which to load encoder')

# Whether to just work with subset of data
parser.add_argument('--propn_sample', type = float,
                    help='proportion of dataset to use',
                    default = 1.0)
parser.add_argument('--propn_labeled', type = float, default = 0.1,
                    help = 'proportion of training data labeled')

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outdir)

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
                 batch_size=64,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=64,
                shuffle=False)

for batch_idx, d in enumerate(train_loader_labeled):
    data_labeled = d
    break

print('num_train_labeled: ', train_set_labeled.num_images)
print('check: \n', data_labeled['image'].shape[0])

print('num_train_unlabeled: \n', train_set_unlabeled.num_images)

print('num_test: ', test_set.num_images)

# SET UP VAE
slen = train_set_unlabeled[0]['image'].shape[0]
latent_dim = 5 # args.latent_dim
n_classes = 10
vae = mnist_vae_lib.HandwritingVAE(latent_dim = latent_dim,
                            n_classes = n_classes,
                            slen = slen)
vae.to(device)
if args.load_classifier:
    print('initializing classifier from ', args.classifier_init)

    vae.classifier.load_state_dict(torch.load(args.classifier_init,
                                    map_location=lambda storage, loc: storage))

print('training vae')

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)

def train_classifier(vae, images, labels, optimizer):
    images = images.to(device)
    labels = labels.to(device)

    for i in range(args.epochs):
        optimizer.zero_grad()

        log_q = vae.classifier(images)

        cross_entropy = vae.get_class_label_cross_entropy(log_q, labels)

        cross_entropy.backward()

        optimizer.step()

        # get classification accuracy:
        z_ind = torch.argmax(log_q.detach(), dim = 1)

        print('accuracy labeled: {:.4g};'.format(torch.mean((z_ind == labels).float())))
        # print('accuracy unlabeled: {:.4g};'.format(mnist_vae_lib.eval_classification_accuracy(vae.classifier, train_loader_unlabeled)))
        print('accuracy test: {:.4g}; \n \n '.format(mnist_vae_lib.eval_classification_accuracy(vae.classifier, test_loader)))

optimizer = optim.Adam([
        {'params': vae.classifier.parameters(), 'lr': args.learning_rate}],
        weight_decay=args.weight_decay)

train_classifier(vae, data_labeled['image'], data_labeled['label'], optimizer)

outfile_final = args.outdir + args.outfilename + '_classifier_final'
print("writing the classifier parameters to " + outfile_final + '\n')
torch.save(vae.classifier.state_dict(), outfile_final)

print('done. Total time: {}secs'.format(time.time() - t0_train))
