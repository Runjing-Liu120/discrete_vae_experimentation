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


def sample_class_weights(class_weights):
    # draw a sample from Categorical variable with
    # probabilities class_weights

    cat_rv = Categorical(probs = class_weights)
    return cat_rv.sample().detach()

def get_concentrated_mask(class_weights, alpha, topk):
    # returns a logical mask, binary for class_weights > alpha
    # AND if class_weights one of the k largest.

    # NOTE: this only works for a vector of class_weights at the moment.

    # boolean vector for where class_weights > alpha
    mask_alpha = (class_weights > alpha).float().detach()

    # but if there are more than k, only take the topk
    mask_topk = torch.zeros(len(class_weights))
    if topk > 0:
        _, topk_domain = torch.topk(class_weights, topk)
        mask_topk[topk_domain] = 1

    return mask_alpha * mask_topk


class PartialMarginalizationREINFORCE(object):
    def __init__(self, f_z,
                        var_params,
                        get_log_q_from_var_params):

        self.f_z = f_z

        self.set_var_params(var_params)

        self.get_log_q_from_var_params = \
                    get_log_q_from_var_params

    def set_var_params(self, var_params):
        self.var_params = deepcopy(var_params)

    def get_log_q(self):
        log_q = self.get_log_q_from_var_params(self.var_params)
        self.class_weights = torch.exp(log_q.detach())
        return log_q

    def get_full_loss(self, diffble = False):
        log_q = self.get_log_q()

        if diffble:
            class_weights = torch.exp(log_q)
        else:
            class_weights = self.class_weights

        full_loss = 0.0
        for i in range(len(class_weights)):
            full_loss = full_loss + class_weights[i] * self.f_z(i)

        return full_loss

    def get_reinforce_grad_sample(self, i, log_q, use_baseline = False):
        if use_baseline:
            z_sample2 = sample_class_weights(self.class_weights)
            baseline = self.f_z(z_sample2)
        else:
            baseline = 0.0

        return (self.f_z(i) - baseline) * log_q[i]

    def get_partial_marginal_loss(self, alpha, topk, use_baseline = False):
        # class weights from the variational distribution
        log_q = self.get_log_q()

        # this is the indicator C_\alpha
        concentrated_mask = get_concentrated_mask(self.class_weights, alpha, topk)
        concentrated_mask = concentrated_mask.float().detach()

        # the summed term
        summed_term = torch.Tensor([0.])
        summed_term.requires_grad_(True)
        for i in range(len(concentrated_mask)):
            if concentrated_mask[i] == 1:
                summed_term = summed_term + self.get_reinforce_grad_sample(i, log_q, use_baseline) * self.class_weights[i]

        # sampled term
        sampled_weight = torch.sum(self.class_weights * (1 - concentrated_mask))
        if torch.sum(1 - concentrated_mask) > 0:
            conditional_class_weights = \
                self.class_weights * (1 - concentrated_mask) / sampled_weight
            conditional_z_sample = sample_class_weights(conditional_class_weights)

            # just for my own sanity ...
            assert (1 - concentrated_mask)[conditional_z_sample] == 1.

            sampled_term = self.get_reinforce_grad_sample(conditional_z_sample, log_q, use_baseline)
        else:
            sampled_term = 0.

        return sampled_term * sampled_weight + summed_term

    def run_SGD(self, init_var_params, alpha, topk, lr = 1.0, n_steps = 10000, use_baseline = False):
        self.set_var_params(init_var_params)
        init_loss = self.get_full_loss()

        # set up optimizer
        params = [self.var_params]
        optimizer = optim.SGD(params, lr = lr)

        loss_array = np.zeros(n_steps + 1)
        phi_array = np.zeros(n_steps + 1)

        loss_array[0] = init_loss.detach().numpy()
        phi_array[0] = init_var_params.detach().numpy()

        for i in range(n_steps):
            # run gradient descent
            optimizer.zero_grad()
            # ps_loss = self.get_partial_marginal_loss(alpha, topk)
            ps_loss = self.get_partial_marginal_loss(alpha, topk, use_baseline)
            ps_loss.backward()
            optimizer.step()

            # save losses
            loss = self.get_full_loss()
            loss_array[i + 1] = loss.detach().numpy()
            phi_array[i + 1] = self.var_params.detach().numpy()

        return loss_array, phi_array
