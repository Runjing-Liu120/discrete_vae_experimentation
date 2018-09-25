import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import timeit

from copy import deepcopy

import itertools

softmax = nn.Softmax(dim = 0)

# The variational distribution for the class labels
class GMMEncoder(nn.Module):
    def __init__(self, data_dim, n_classes, hidden_dim = 50):

        super(GMMEncoder, self).__init__()

        # image / model parameters
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # define the linear layers
        self.fc1 = nn.Linear(self.data_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_classes)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, image):

        # feed through neural network
        h = image.view(-1, self.data_dim)

        h = F.leaky_relu(self.fc1(h))
        h = F.leaky_relu(self.fc2(h))
        class_weights = self.softmax(self.fc3(h))

        return class_weights

def get_normal_loglik(x, mu, sigma):
    return - (x - mu)**2 / (2 * sigma ** 2) - torch.log(sigma)

class GMMExperiments(object):
    def __init__(self, n_obs, mu0, sigma0, n_clusters, hidden_dim = 200):

        # dimension of the problem
        self.dim = len(mu0)
        self.n_clusters = n_clusters

        # prior parameters
        self.mu0 = mu0
        self.sigma0 = torch.Tensor([sigma0])
        # uniform prior on weights
        self.prior_weights = torch.ones(self.n_clusters) / self.n_clusters

        # true parameters
        self.set_true_params()
        self.cat_rv = Categorical(probs = self.prior_weights)

        # the encoder
        self.gmm_encoder = GMMEncoder(data_dim = self.dim,
                             n_classes = self.n_clusters,
                             hidden_dim = hidden_dim)

        self.var_params = {'encoder_params': self.gmm_encoder.parameters()}

        # other variational paramters: we use point masses for
        # the means and variances
        self.set_random_var_params()

        # draw data
        self.n_obs = n_obs
        self.y, self.z = self.draw_data(n_obs = n_obs)

    def set_var_params(self, init_mu, init_sigma):
        self.var_params['centroids'] = init_mu
        self.var_params['sigma'] = init_sigma

    def set_random_var_params(self):
        init_mu = torch.randn((self.n_clusters, self.dim)) * self.sigma0 + self.mu0
        init_sigma = torch.rand(1)

        self.set_var_params(init_mu, init_sigma)

    def set_true_params(self):
        # draw means from the prior
        # each row is a cluster mean
        self.true_mus = torch.randn((self.n_clusters, self.dim)) * self.sigma0 + self.mu0

        # just set a data variance
        self.true_sigma = 1.0

    def draw_data(self, n_obs = 1):

        y = torch.zeros((n_obs, self.dim))
        z = torch.zeros(n_obs)
        for i in range(n_obs):
            # class belonging
            z_sample = self.cat_rv.sample()
            z[i] = z_sample

            # observed data
            y[i, :] = self.true_mus[z_sample, :] + torch.randn(2) * self.true_sigma

        # some indices we cache and use later
        self.seq_tensor = torch.LongTensor([i for i in range(n_obs)])

        return y, z

    def get_log_class_weights(self):
        self.log_class_weights = torch.log(self.gmm_encoder.forward(self.y))
        return self.log_class_weights

    def _get_centroid_mask(self, z):
        mask = torch.zeros((self.n_obs, self.n_clusters))
        mask[self.seq_tensor, z] = 1

        return mask

    def get_loss_conditional_z(self, z):
        centroids = self.var_params['centroids']
        sigma = self.var_params['sigma']

        centroid_mask = self._get_centroid_mask(z)
        centroids_masked = torch.matmul(centroid_mask, centroids)

        loglik_z = get_normal_loglik(self.y, centroids_masked, sigma).sum(dim = 1)

        mu_prior_term = get_normal_loglik(centroids, self.mu0, self.sigma0).sum()

        z_prior_term = 0.0 # torch.log(self.prior_weights[z])

        z_entropy_term = (- torch.exp(self.log_class_weights) * self.log_class_weights).sum()

        return - (loglik_z + mu_prior_term + z_prior_term + z_entropy_term)
