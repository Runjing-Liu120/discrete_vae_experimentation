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
    mask_alpha = (class_weights >= alpha).float().detach()

    # but if there are more than k, only take the topk
    mask_topk = torch.zeros(class_weights.shape)

    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])

    if topk > 0:
        _, topk_domain = torch.topk(class_weights, topk)
        # mask_topk[topk_domain] = 1
        # print(topk_domain)
        for i in range(topk):
            mask_topk[seq_tensor, topk_domain[:, i]] = 1

    return mask_alpha * mask_topk


class PartialMarginalizationREINFORCE(object):
    def __init__(self, experiment_class):

        self.experiment_class = experiment_class

    # def set_var_params(self, var_params):
    #     self.experiment_class.var_params = var_params

    def set_and_get_log_q(self):
        log_q = self.experiment_class.get_log_q()
        self.class_weights = torch.exp(log_q.detach())

        return log_q

    # def get_full_loss(self, diffble = False):
    #     log_q = self.get_log_q()
    #
    #     if diffble:
    #         class_weights = torch.exp(log_q)
    #     else:
    #         class_weights = self.class_weights
    #
    #     full_loss = 0.0
    #     for i in range(len(class_weights)):
    #         full_loss = full_loss + class_weights[i] * self.f_z(i)
    #
    #     return full_loss

    def get_reinforce_grad_sample(self, f_z_i, log_q_i, use_baseline = False):
        if use_baseline:
            z_sample2 = sample_class_weights(self.class_weights)
            baseline = self.experiment_class.f_z(z_sample2)
        else:
            baseline = 0.0

        return (f_z_i - baseline) * log_q_i

    def get_partial_marginal_loss(self, alpha, topk,
                                    use_baseline = False,
                                    return_full_loss = True):
        # class weights from the variational distribution
        log_q = self.set_and_get_log_q()

        # this is the indicator C_\alpha
        # print('class_weights', self.class_weights)
        concentrated_mask = get_concentrated_mask(self.class_weights, alpha, topk)
        concentrated_mask = concentrated_mask.float().detach()

        # the summed term
        summed_term = torch.Tensor([0.])
        summed_term.requires_grad_(True)

        full_loss = 0.0

        for i in range(concentrated_mask.shape[1]):

            f_z_i = self.experiment_class.f_z(i)
            log_q_i = log_q[:, i]
            # print('f', f_z_i)
            summed_term_ = \
                (self.get_reinforce_grad_sample(f_z_i, log_q_i, use_baseline) * \
                self.class_weights[:, i] * concentrated_mask[:, i]).sum()

            summed_term = summed_term + summed_term_

            if return_full_loss:
                full_loss = full_loss + (torch.exp(log_q_i) * f_z_i).sum()

        # sampled term
        sampled_weight = torch.sum(self.class_weights * (1 - concentrated_mask), dim = 1, keepdim = True)
        # print('concentrated_mask', concentrated_mask)
        # if torch.sum(1 - concentrated_mask) > 0:
        if not(topk == self.class_weights.shape[1]): 
            conditional_class_weights = \
                self.class_weights * (1 - concentrated_mask) / (sampled_weight)

            conditional_z_sample = sample_class_weights(conditional_class_weights)

            # just for my own sanity ...
            seq_tensor = torch.LongTensor([i for i in range(self.class_weights.shape[0])])
            assert np.all((1 - concentrated_mask)[seq_tensor, conditional_z_sample].numpy() == 1.)

            f_z_i_sample = self.experiment_class.f_z(conditional_z_sample)
            log_q_i_sample = log_q[:, conditional_z_sample]
            sampled_term = self.get_reinforce_grad_sample(f_z_i_sample, log_q_i_sample, use_baseline)

        else:
            sampled_term = 0.

        return (sampled_term * sampled_weight).sum() + summed_term, full_loss

    def run_SGD(self, alpha, topk,
                    lr = 1.0, n_steps = 10000, use_baseline = False,
                    use_true_grad = False):

        _, init_loss = self.get_partial_marginal_loss(alpha, topk, use_baseline)

        # set up optimizer
        # params = [self.experiment_class.var_params]
        params = [{'params': self.experiment_class.var_params[key]} for key in self.experiment_class.var_params]

        optimizer = optim.SGD(params, lr = lr)

        loss_array = np.zeros(n_steps + 1)
        # phi_array = np.zeros(n_steps + 1)

        loss_array[0] = init_loss.detach().numpy()
        # phi_array[0] = init_var_params.detach().numpy()

        for i in range(n_steps):
            # run gradient descent
            optimizer.zero_grad()
            # ps_loss = self.get_partial_marginal_loss(alpha, topk)
            ps_loss, loss = self.get_partial_marginal_loss(alpha, topk, use_baseline)

            if use_true_grad:
                loss.backward()
            else:
                ps_loss.backward()

            optimizer.step()

            # save losses
            # loss = self.get_full_loss()
            loss_array[i + 1] = loss.detach().numpy()
            # phi_array[i + 1] = self.experiment_class.var_params.detach().numpy()

        var_param_opt = self.experiment_class.var_params

        return var_param_opt, loss_array
