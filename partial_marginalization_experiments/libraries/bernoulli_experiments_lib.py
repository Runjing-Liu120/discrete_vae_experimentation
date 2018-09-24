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
sigmoid = nn.Sigmoid()

def get_bernoulli_log_prob(e_b, draw):
    return torch.log(e_b + 1e-12) * torch.sum(draw) + \
                torch.log(1 - e_b + 1e-12) * torch.sum(1 - draw)

# class to run Bernoulli Experiments
class BernoulliExperiments(object):
    def __init__(self, p0, dim):
        self.p0 = p0
        self.dim = dim

        self.set_draw_array()

    def set_draw_array(self):
        # defines the 2**d vector of possible combinations

        self.draw_array = []
        i = 0
        for draw in itertools.product(range(2), repeat=self.dim):
            draw_tensor = torch.Tensor(draw)
            self.draw_array.append(draw_tensor)

    def get_loss_from_draw_i(self, i):
        # returns the loss for the ith entry in draw array
        return torch.sum((self.draw_array[i] - self.p0) ** 2)

    def get_bernoulli_log_prob_vec(self, phi):
        # returns a vector of log probabilities for all the possible draws
        log_probs = torch.zeros(len(self.draw_array))

        e_b = sigmoid(phi)

        for i in range(len(self.draw_array)):
            draw_tensor = torch.Tensor(self.draw_array[i])
            log_probs[i] = get_bernoulli_log_prob(e_b, draw_tensor)

        return log_probs

    def get_bernoulli_prob_vec(self, phi):
        # returns a vector of probabilities for all the possible draws
        return torch.exp(self.get_bernoulli_log_prob_vec(phi))

    def get_bernoulli_log_prob_i(self, phi, i):
        # returns the log probabilitie for draw i
        e_b = sigmoid(phi)
        return get_bernoulli_log_prob(e_b, self.draw_array[i])

    def get_sampled_reinforce_ps_loss(self, phi, i, concentrated_mask):
        # after drawing i, compute the reinforce pseudoloss
        return self.get_bernoulli_log_prob_i(phi, i) * \
                            self.get_loss_from_draw_i(i) * \
                            (concentrated_mask[i] == 0.).float().detach()

    def get_sampled_reinforce_cv_ps_loss(self, phi, i, concentrated_mask):
        # after drawing i, compute the reinforce pseudoloss
        # with control variate

        # compute control variate
        bern_probs = self.get_bernoulli_prob_vec(phi)
        cat_rv = Categorical(probs = bern_probs)
        z_sample = cat_rv.sample().detach()
        cv = self.get_loss_from_draw_i(z_sample) * (concentrated_mask[z_sample] == 0.).float().detach()

        # the actual loss at i
        f_z = self.get_loss_from_draw_i(i) * (concentrated_mask[i] == 0.).float().detach()

        return self.get_bernoulli_log_prob_i(phi, i) * (f_z - cv).detach()
