import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import sys
sys.path.insert(0, '../libraries/')

from  common_utils import get_bernoulli_loglik

import timeit

from copy import deepcopy

import itertools

softmax = nn.Softmax(dim = 0)


class NonLinearEncoder(nn.Module):
    def __init__(self, latent_dim = 200,
                    slen = 28):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(NonLinearEncoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.latent_dim)

        self.sigmoid = nn.Sigmoid()


    def forward(self, image):

        # feed through neural network
        h = image.view(-1, self.n_pixels)

        h = F.leaky_relu(self.fc1(h))
        h = F.leaky_relu(self.fc2(h))
        latent_mean = self.sigmoid(self.fc3(h))

        return latent_mean

class NonLinearDecoder(nn.Module):
    def __init__(self, latent_dim = 200,
                    slen = 28):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(NonLinearDecoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen

        # define the linear layers
        self.fc1 = nn.Linear(self.latent_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.n_pixels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_vars):

        # feed through neural network
        assert latent_vars.shape[1] == self.latent_dim

        h = F.leaky_relu(self.fc1(latent_vars))
        h = F.leaky_relu(self.fc2(h))
        image_mean = self.sigmoid(self.fc3(h)).view(-1, self.slen, self.slen)

        return image_mean

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(VAE, self).__init__()

        assert encoder.latent_dim == decoder.latent_dim

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, image):
        latent_means = self.encoder.forward(image)

        bernoulli_rv = Bernoulli(latent_means)
        bernoulli_samples = bernoulli_rv.sample().detach()

        image_mean = self.decoder.forward(bernoulli_samples)

        return image_mean, latent_means, bernoulli_samples

    def loss(self, image):
        # note this loss is not differentiable because of the
        # Bernoulli samples

        image_mean, latent_means, bernoulli_samples = self.forward(image)

        neg_log_lik = -get_bernoulli_loglik(image_mean, image)

        kl_latent_array = bernoulli_samples * \
                            np.log(2 * bernoulli_samples + 1e-8) + \
                    (1 - bernoulli_samples) * \
                            np.log(2 * (1 - bernoulli_samples) + 1e-8)
        kl_latent = kl_latent_array.sum(1)

        return neg_log_lik + kl_latent



#
