import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import common_utils as common_utils

import timeit

from copy import deepcopy

import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_concentrated_mask(class_weights, alpha, topk):
    # returns a logical mask, binary for class_weights > alpha
    # AND if class_weights one of the k largest.

    # NOTE: this only works for a vector of class_weights at the moment.

    # boolean vector for where class_weights > alpha
    # mask_alpha = (class_weights >= alpha).float().detach()

    # but if there are more than k, only take the topk
    mask_topk = torch.zeros(class_weights.shape).to(device)

    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])

    if topk > 0:
        _, topk_domain = torch.topk(class_weights, topk)
        # mask_topk[topk_domain] = 1
        # print(topk_domain)
        for i in range(topk):
            mask_topk[seq_tensor, topk_domain[:, i]] = 1
    else:
        topk_domain = None

    return mask_topk, topk_domain, seq_tensor

def get_full_loss(f_z, class_weights):

    full_loss = 0.0
    for i in range(class_weights.shape[1]):
        full_loss = full_loss + class_weights[:, i] * f_z(i)

    return full_loss.sum()

def get_partial_marginal_loss(f_z, log_q, alpha, topk,
                                use_baseline = True,
                                use_term_one_baseline = True):

    # class weights from the variational distribution
    assert np.all(log_q.detach().cpu().numpy() <= 0)
    class_weights = torch.exp(log_q.detach())
    # assert np.all(np.abs(class_weights.cpu().sum(1).numpy() - 1.0) < 1e-6), \
    #             np.max(np.abs(class_weights.cpu().sum(1).numpy() - 1.0))

    # this is the indicator C_\alpha
    # NOTE:
    assert topk == 0
    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])
    # concentrated_mask, topk_domain, seq_tensor = \
    #     get_concentrated_mask(class_weights, alpha, topk)
    # concentrated_mask = concentrated_mask.float().detach()

    # the summed term
    summed_term = 0.0

    for i in range(topk):
        # NOTE: remove this
        assert 1 == 2
        
        summed_indx = topk_domain[:, i]
        f_z_i = f_z(summed_indx)
        assert len(f_z_i) == log_q.shape[0]

        log_q_i = log_q[seq_tensor, summed_indx]

        if (use_term_one_baseline) and (use_baseline):
            # print('using term 1 baseline')
            z_sample2 = common_utils.sample_class_weights(class_weights)
            baseline = f_z(z_sample2).detach()

        else:
            baseline = 0.0

        reinforce_grad_sample = \
                common_utils.get_reinforce_grad_sample(f_z_i, log_q_i, baseline)

        summed_term = summed_term + \
                        ((reinforce_grad_sample + f_z_i) * \
                        class_weights[seq_tensor, summed_indx].squeeze()).sum()

    # sampled term
    # NOTE:
    # sampled_weight = torch.sum(class_weights * (1 - concentrated_mask), dim = 1, keepdim = True)
    sampled_weight = torch.ones((class_weights.shape[0], 1))
    if not(topk == class_weights.shape[1]):
        # NOTE:
        # conditional_class_weights = \
        #     class_weights * (1 - concentrated_mask) / (sampled_weight)

        # conditional_z_sample = common_utils.sample_class_weights(conditional_class_weights)
        conditional_z_sample = common_utils.sample_class_weights(class_weights)
        # print(conditional_z_sample)

        # just for my own sanity ...
        # assert np.all((1 - concentrated_mask)[seq_tensor, conditional_z_sample].cpu().numpy() == 1.), \
        #             'sampled_weight {}'.format(sampled_weight)

        f_z_i_sample = f_z(conditional_z_sample)
        log_q_i_sample = log_q[seq_tensor, conditional_z_sample]

        if use_baseline:
            if not use_term_one_baseline:
                # print('using alt. covariate')
                # sample from the conditional distribution instead
                z_sample2 = common_utils.sample_class_weights(conditional_class_weights)
                baseline2 = f_z(z_sample2).detach()

            else:
                z_sample2 = common_utils.sample_class_weights(class_weights)
                baseline2 = f_z(z_sample2).detach()
        else:
            baseline2 = 0.0

        sampled_term = common_utils.get_reinforce_grad_sample(f_z_i_sample,
                                    log_q_i_sample, baseline2) + f_z_i_sample

    else:
        sampled_term = 0.

    return (sampled_term * sampled_weight.squeeze()).sum() + summed_term



# class PartialMarginalizationREINFORCE(object):
#     def __init__(self, experiment_class):
#
#         self.experiment_class = experiment_class
#
#         _ = self.set_and_get_log_q()
#
#         # cache this we will need it later
#         self.seq_tensor = torch.LongTensor([i for i in range(self.class_weights.shape[0])])
#
#     # def set_var_params(self, var_params):
#     #     self.experiment_class.var_params = var_params
#
#     def set_and_get_log_q(self):
#         log_q = self.experiment_class.get_log_q()
#         return log_q
#
#     # def get_full_loss(self, diffble = False):
#     #     log_q = self.get_log_q()
#     #
#     #     if diffble:
#     #         class_weights = torch.exp(log_q)
#     #     else:
#     #         class_weights = self.class_weights
#     #
#     #     full_loss = 0.0
#     #     for i in range(len(class_weights)):
#     #         full_loss = full_loss + class_weights[i] * self.f_z(i)
#     #
#     #     return full_loss


    # def run_SGD(self, lo
    #                 lr = 1.0, n_steps = 10000,
    #                 use_true_grad = False):
    #
    #     _, init_loss = self.get_partial_marginal_loss(alpha, topk, use_baseline)
    #
    #     # set up optimizer
    #     # params = [self.experiment_class.var_params]
    #     params = [{'params': self.experiment_class.var_params[key]} for key in self.experiment_class.var_params]
    #
    #     optimizer = optim.SGD(params, lr = lr)
    #
    #     loss_array = np.zeros(n_steps + 1)
    #     # phi_array = np.zeros(n_steps + 1)
    #
    #     loss_array[0] = init_loss.detach().numpy()
    #     # phi_array[0] = init_var_params.detach().numpy()
    #
    #     for i in range(n_steps):
    #         # run gradient descent
    #         optimizer.zero_grad()
    #         # ps_loss = self.get_partial_marginal_loss(alpha, topk)
    #         ps_loss, loss = self.get_partial_marginal_loss(alpha, topk, use_baseline)
    #
    #         if use_true_grad:
    #             loss.backward()
    #         else:
    #             ps_loss.backward()
    #
    #         optimizer.step()
    #
    #         # save losses
    #         # loss = self.get_full_loss()
    #         loss_array[i + 1] = loss.detach().numpy()
    #         # phi_array[i + 1] = self.experiment_class.var_params.detach().numpy()
    #
    #     var_param_opt = self.experiment_class.var_params
    #
    #     return var_param_opt, loss_array
