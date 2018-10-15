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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_class_weights(class_weights):
    # draw a sample from Categorical variable with
    # probabilities class_weights

    cat_rv = Categorical(probs = class_weights)
    return cat_rv.sample().detach()

def run_SGD(get_loss, params,
                lr = 1.0, n_steps = 10000,
                get_full_loss = None):

    # get_loss should be a function that returns a ps_loss such that
    # ps_loss.backward() returns an unbiased estimate of the gradient.
    # in general, ps_loss might not equal the actual loss.

    # set up optimizer
    params_list = [{'params': params[key]} for key in params]
    optimizer = optim.SGD(params_list, lr = lr)

    loss_array = np.zeros(n_steps)

    for i in range(n_steps):
        # run gradient descent
        optimizer.zero_grad()
        # ps_loss = self.get_partial_marginal_loss(alpha, topk)
        loss = get_loss()
        loss.backward()
        optimizer.step()

        # save losses
        if get_full_loss is not None:
            full_loss = get_full_loss()
        else:
            full_loss = loss

        loss_array[i] = full_loss.detach().numpy()
        # phi_array[i + 1] = self.experiment_class.var_params.detach().numpy()

    opt_params = params

    return loss_array, opt_params
