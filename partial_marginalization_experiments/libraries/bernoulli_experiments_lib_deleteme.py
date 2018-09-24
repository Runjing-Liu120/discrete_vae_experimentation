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

## Losses

def get_loss(phi, p0):
    # loss for bernoulli experiments
    sum_p0 = torch.sum(p0**2)
    sum_1mp = torch.sum((1 - p0)**2)

    e_b = sigmoid(phi)

    return torch.sum(sum_p0 * (1 - e_b) + sum_1mp * e_b)

def get_bernoulli_log_prob(e_b, draw):
    return torch.log(e_b + 1e-12) * torch.sum(draw) + \
                torch.log(1 - e_b + 1e-12) * torch.sum(1 - draw)

def get_reinforce_ps_loss(phi, p0, reinforce = False):
    # returns pseudoloss: loss whose gradient is unbiased for the
    # true gradient

    d = len(p0)
    e_b = sigmoid(phi)

    bn_rv = Bernoulli(probs = torch.ones(d) * e_b)
    binary_samples = bn_rv.sample().detach()
    # binary_samples = (torch.rand(d) > e_b).float().detach()

    if reinforce:
        binary_samples_ = bn_rv.sample().detach()
        baseline = torch.sum((binary_samples_ - p0)**2)

    else:
        baseline = 0.0

    sampled_loss = torch.sum((binary_samples - p0)**2)

    # probs, draw_array = get_all_probs(e_b, d)
    # losses_array = get_losses_from_draw_array(draw_array, p0)
    #
    # cat_rv = Categorical(probs)
    # indx = cat_rv.sample()
    # binary_samples = draw_array[indx]
    # sampled_loss = losses_array[indx]
    #
    sampled_log_q = get_bernoulli_log_prob(e_b, binary_samples)

    ps_loss = (sampled_loss - baseline).detach() * sampled_log_q

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

# Functions for our classic/reinforce mix
def get_all_probs(e_b, d):
    # returns a 2^d length vector of probabilities for all possible combinations

    log_probs = torch.zeros(2 ** d)
    draw_array = []
    i = 0
    for draw in itertools.product(range(2), repeat=d):
        draw_tensor = torch.Tensor(draw)
        log_probs[i] = get_bernoulli_log_prob(e_b, draw_tensor)

        draw_array.append(draw_tensor)

        i += 1

    return torch.exp(log_probs), draw_array



def get_losses_from_draw_array(draw_array, p0):
    return torch.Tensor([torch.sum((draw - p0)**2) for draw in draw_array])

def get_mixed_reinforce_ps_loss(phi, p0, num_reinforced):

    d = len(p0)

    e_b = sigmoid(phi)

    # get probabilities for all 2^d combinations
    probs, draw_array = get_all_probs(e_b, d)

    # get losses for all 2^d combinations
    losses_array = get_losses_from_draw_array(draw_array, p0).detach()

    # sample from conditional probabiltiies
    z_sample, sampled_z_domain, sampled_weight, \
            unsampled_z_domain, unsampled_weight = \
                sample_class_weights(probs, num_reinforced)

    # contribution of the unsampled weights
    unsample_mask = torch.zeros(len(losses_array))
    unsample_mask[unsampled_z_domain] = 1
    # print(unsample_mask)
    unsampled_loss = torch.sum(probs * \
                                losses_array * unsample_mask.detach())

    # contribution of the sample weights
    sample_mask = torch.zeros(len(losses_array))
    sample_mask[z_sample] = 1.0
    sample_mask = sample_mask.detach()

    sum_unsampled_weights = torch.sum(probs * unsample_mask)

    torch.sum(losses_array * torch.log((probs + 1e-6))) * sample_mask

    # sampled_loss = (1 - sum_unsampled_weights) * \
    #                     (torch.sum(losses_array * sample_mask)).detach() + \
    #             (1 - sum_unsampled_weights).detach() * \
    #                     torch.sum(losses_array * \
    #                               torch.log((probs + 1e-6) /  (1 - sum_unsampled_weights + 1e-6)) * \
    #                               sample_mask)

    # print(sampled_loss)
    # print((1 - sum_unsampled_weights) * torch.sum(losses_array * sample_mask))
    # print(torch.sum(torch.log(probs / torch.sum(sampled_weight)) * sample_mask))
    # print(torch.log(probs[z_sample] / torch.sum(sampled_weight)))

    return unsampled_loss + sampled_loss

#     return torch.sum(probs * losses_array)



## Training functions
def run_SGD(phi0, p0, get_ps_loss_fun, lr = 1.0, n_steps = 10000):

    init_loss = get_loss(phi0, p0)

    # set up optimizer
    phi = deepcopy(phi0)
    params = [phi]
    optimizer = optim.SGD(params, lr = lr)

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
