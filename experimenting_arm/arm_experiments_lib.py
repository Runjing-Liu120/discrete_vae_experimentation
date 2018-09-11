import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical
import torch.nn.functional as F

import timeit

from copy import deepcopy

softmax = nn.Softmax(dim = 0)
sigmoid = nn.Sigmoid()

## Losses

def get_loss(phi, p0):
    # loss for bernoulli experiments
    sum_p0 = torch.sum(p0**2)
    sum_1mp = torch.sum((1 - p0)**2)

    e_b = sigmoid(phi)

    return torch.sum(sum_p0 * (1 - e_b) + sum_1mp * e_b)

def get_reinforce_ps_loss(phi, p0):
    # returns pseudoloss: loss whose gradient is unbiased for the
    # true gradient

    d = len(p0)
    e_b = sigmoid(phi)

    binary_samples = (torch.rand(d) > e_b).float().detach()

    sampled_loss = torch.sum((binary_samples - p0)**2)

    sampled_log_q = torch.log(e_b + 1e-12) * torch.sum(binary_samples) + \
                        torch.log(1 - e_b + 1e-12) * torch.sum(1 - binary_samples)

    ps_loss = sampled_loss.detach() * sampled_log_q

    return ps_loss

def get_arm_ps_loss(phi, p0):
    # returns a value whose gradient (wrt to phi) is the ARM gradient
    # estimator

    d = len(p0)

    sigma_phi = sigmoid(phi).detach()
    sigma_neg_phi = sigmoid(-phi).detach()

    # sample uniform
    u = torch.rand(d)

    # compute z1 and z2
    z1 = (u > sigma_neg_phi).float()
    z2 = (u < sigma_phi).float()

    # compute gradient:
    f_delta = torch.sum((z1 - p0)**2) - torch.sum((z2 - p0)**2)
    grad_f = f_delta * torch.sum(u - 0.5)

    return grad_f.detach() * phi

## Training functions
def run_SGD(phi0, p0, get_ps_loss_fun, lr = 1.0, n_steps = 10000):

    init_loss = get_loss(phi0, p0)

    # set up optimizer
    phi = deepcopy(phi0)
    params = [phi]
    optimizer = optim.SGD(params, lr = 1)

    loss_array = np.zeros(n_steps + 1)
    phi_array = np.zeros(n_steps + 1)

    loss_array[0] = init_loss.detach().numpy()
    phi_array[0] = phi0.detach().numpy()

    for i in range(n_steps):
        # run gradient descent
        optimizer.zero_grad()
        ps_loss = get_ps_loss_fun(phi, p0)
        ps_loss.backward()
        optimizer.step()

        # save losses
        loss = get_loss(phi.detach(), p0)
        loss_array[i + 1] = loss.detach().numpy()
        phi_array[i + 1] = phi.detach().numpy()

    return loss_array, phi_array
