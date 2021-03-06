import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical
import torch.nn.functional as F

import mnist_utils
import timeit

import sys
sys.path.insert(0, '../../partial_marginalization_experiments/libraries/')
import partial_marginalization_lib as pm_lib
import common_utils as pm_common_utils

import semisupervised_vae_lib as ss_vae_lib

from torch.optim import lr_scheduler

from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28,
                    n_classes = 10):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen
        self.n_classes = n_classes

        # define the linear layers
        # self.fc1 = nn.Linear(self.n_pixels + self.n_classes, 256) # 128 hidden nodes; two more layers
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, latent_dim * 2)

        self.fc1 = nn.Linear(self.n_pixels + self.n_classes, 256) # 128 hidden nodes; two more layers
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, latent_dim * 2)


    def forward(self, image, z):

        # feed through neural network
        h = image.view(-1, self.n_pixels)
        h = torch.cat((h, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        h = self.fc4(h)
        # h = self.fc4(h)
        # h = self.fc5(h)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim
        # indx3 = 2 * self.latent_dim + self.n_classes

        latent_means = h[:, 0:indx1]
        latent_std = torch.exp(h[:, indx1:indx2])
        # free_class_weights = h[:, indx2:]
        # class_weights = mnist_utils.get_symplex_from_reals(free_class_weights)

        return latent_means, latent_std #, class_weights


class Classifier(nn.Module):
    def __init__(self, slen = 28, n_classes = 10):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()

        self.slen = slen
        self.n_pixels = slen ** 2
        self.n_classes = n_classes

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_classes)

        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        return self.log_softmax(h)

class BaselineLearner(nn.Module):
    def __init__(self, slen = 28):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(BaselineLearner, self).__init__()

        self.slen = slen
        self.n_pixels = slen ** 2

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h

class MLPConditionalDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                        n_classes = 10,
                        slen = 28):

        # This takes the latent parameters and returns the
        # mean and variance for the image reconstruction

        super(MLPConditionalDecoder, self).__init__()

        # image/model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        self.fc1 = nn.Linear(latent_dim + n_classes, 128)
        self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(256, self.n_pixels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_params, z):
        assert latent_params.shape[1] == self.latent_dim
        assert z.shape[1] == self.n_classes # z should be one hot encoded
        assert latent_params.shape[0] == z.shape[0]

        h = torch.cat((latent_params, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        h = self.fc4(h)

        h = h.view(-1, self.slen, self.slen)

        # image_mean = h[:, 0, :, :]
        # image_std = torch.exp(h[:, 1, :, :])
        image_mean = self.sigmoid(h)

        return image_mean # , image_std

class MNISTConditionalVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super(MNISTConditionalVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder.latent_dim == self.decoder.latent_dim
        assert self.encoder.n_classes == self.decoder.n_classes
        assert self.encoder.slen == self.decoder.slen

        # save some parameters
        self.latent_dim = self.encoder.latent_dim
        self.n_classes = self.encoder.n_classes
        self.slen = self.encoder.slen

    def forward(self, image, z):

        one_hot_z = mnist_utils.get_one_hot_encoding_from_int(z, self.n_classes)

        assert one_hot_z.shape[0] == image.shape[0]
        assert one_hot_z.shape[1] == self.n_classes

        latent_means, latent_std = self.encoder(image, one_hot_z)

        latent_samples = torch.randn(latent_means.shape).to(device) * latent_std + latent_means

        assert one_hot_z.shape[0] == latent_samples.shape[0]
        assert one_hot_z.shape[1] == self.n_classes

        image_mean = self.decoder(latent_samples, one_hot_z)
        image_var = None

        return latent_means, latent_std, latent_samples, image_mean, image_var

def mnist_loglik(image, image_mean, image_var):
    return mnist_utils.get_bernoulli_loglik(image_mean, image)

def get_mnist_vae(latent_dim = 5,
                    n_classes = 10,
                    slen = 28):

    encoder = MLPEncoder(latent_dim = latent_dim,
                            slen = slen,
                            n_classes = n_classes)
    decoder = MLPConditionalDecoder(latent_dim = latent_dim,
                                    slen = slen,
                                    n_classes = n_classes)
    cond_vae = MNISTConditionalVAE(encoder, decoder)

    classifier = Classifier(n_classes = n_classes, slen = slen)

    return ss_vae_lib.SemiSupervisedVAE(cond_vae, classifier, mnist_loglik)
