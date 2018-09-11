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

## Losses

def get_loss(phi, p0):
    # loss for bernoulli experiments
    sum_p0 = torch.sum(p0**2)
    sum_1mp = torch.sum((1 - p0)**2)
    sum_p = torch.Tensor([sum_p0, sum_1mp])

    e_b = softmax(phi)

    return torch.sum(sum_p * e_b)

def get_reinforce_ps_loss(phi, p0):
    # returns pseudoloss: loss whose gradient is unbiased for the
    # true gradient

    d = len(p0)
    e_b = softmax(phi)

    binary_samples = (torch.rand(d) > e_b[0]).float().detach()

    sampled_loss = torch.sum((binary_samples - p0)**2)

    ps_loss = sampled_loss.detach() * torch.sum(
                torch.log(e_b + 1e-8) * \
                torch.Tensor([torch.sum(binary_samples),
                              torch.sum(1 - binary_samples)]).detach())

    return ps_loss

def get_arm_ps_loss(phi, p0):
    # returns a value whose gradient (wrt to phi) is the ARM gradient
    # estimator

    d = len(p0)

    sigma_phi = softmax(phi).detach()
    sigma_neg_phi = softmax(-phi).detach()

    # sample uniform
    u = torch.rand(d)

    # compute z1 and z2
    z1 = (u > sigma_neg_phi).float()
    z2 = (u < sigma_phi).float()

    # compute gradient:
    f_delta = torch.sum((z_1 - p0)**2) - torch.sum((z_2 - p0)**2)
    grad_f = f_delta * torch.sum(u - 0.5)

    return grad_f.detach() * phi

## Training functions
def train_with_true_grad(phi0, p0, lr = 1.0, n_steps = 10000):

    init_loss = get_loss(phi0, p0)

    # set up optimizer
    phi = deepcopy(phi0)
    params = [phi]
    optimizer = optim.SGD(params, lr = 1)

    loss_array = np.zeros(n_steps + 1)
    phi_array = np.zeros((n_steps + 1, 2))

    loss_array[0] = init_loss.detach().numpy()
    phi_array[0, :] = phi0.detach().numpy()

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = get_loss(phi, p0)

        loss.backward()

        optimizer.step()

        loss_array[i + 1] = loss.detach().numpy()
        phi_array[i + 1, :] = phi.detach().numpy()


    return loss_array, phi_array

def train_with_reinforce_grad(phi0, p0, lr = 1.0, n_steps = 10000):

    init_loss = get_loss(phi0, p0)

    # set up optimizer
    phi = deepcopy(phi0)
    params = [phi]
    optimizer = optim.SGD(params, lr = 1)

    loss_array = np.zeros(n_steps + 1)
    phi_array = np.zeros((n_steps + 1, 2))

    loss_array[0] = init_loss.detach().numpy()
    phi_array[0, :] = phi0.detach().numpy()

    for i in range(n_steps):
        optimizer.zero_grad()
        ps_loss = get_reinforce_ps_loss(phi, p0)

        ps_loss.backward()

        optimizer.step()

        loss_array[i+1] = get_loss(phi.detach(), p0).numpy()
        phi_array[i+1, :] = phi.detach().numpy()

    return loss_array, phi_array
