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

def get_importance_sampled_loss(f_z, log_q,
                                importance_weights = None,
                                use_baseline = True):
    # class weights from the variational distribution
    assert np.all(log_q.detach().cpu().numpy() < 0)
    class_weights = torch.exp(log_q.detach())

    assert np.all(np.abs(class_weights.numpy().sum(axis = 1) - 1.0) < 1e-6), \
            np.max(np.abs(class_weights.numpy().sum(axis = 1) - 1.0))

    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])

    # sample from conditional distribution
    if importance_weights is not None:
        assert importance_weights.shape[0] == log_q.shape[0]
        assert importance_weights.shape[1] == log_q.shape[1]
        assert np.all(np.abs(importance_weights.cpu().numpy().sum(axis = 1) - 1.0) < 1e-4),\
                np.max(np.abs(importance_weights.cpu().numpy().sum(axis = 1) - 1.0))

        z_sample = common_utils.sample_class_weights(importance_weights)
        importance_weighting = class_weights[seq_tensor, z_sample] / \
                                    importance_weights[seq_tensor, z_sample]
    else:
        z_sample = common_utils.sample_class_weights(class_weights)
        importance_weighting = 1.0

    f_z_i_sample = f_z(z_sample)
    log_q_i_sample = log_q[seq_tensor, z_sample]

    if use_baseline:
        z_sample2 = common_utils.sample_class_weights(class_weights)
        baseline = f_z(z_sample2).detach()
    else:
        baseline = 0.0

    reinforce_grad_sample = \
        common_utils.get_reinforce_grad_sample(f_z_i_sample, log_q_i_sample, \
                                                baseline) + f_z_i_sample

    return (reinforce_grad_sample * importance_weighting).sum()



#####################
# functions for importance sampling galaxy images
def get_normalized_image(image_batch, attn_offset):
    # normalizes the images to get importance weights

    slen = image_batch.shape[-1]
    image_batch_cropped = image_batch[:, :, attn_offset:(slen - attn_offset), \
                                            attn_offset:(slen - attn_offset)]

    image_batch_cropped_channel_sumed = image_batch_cropped.sum(dim = 1)
    image_batch_cropped_normalized = \
        image_batch_cropped_channel_sumed / \
        image_batch_cropped_channel_sumed.sum(dim = 1, keepdim = True).sum(dim = 2, keepdim = True)

    return image_batch_cropped_normalized

def get_importance_weights(image_batch, attn_offset, prob_off):
    # appends probability of being OFF to the normalized images
    batch_size = image_batch.shape[0]
    normalized_image = get_normalized_image(image_batch, attn_offset)

    importance_weights = normalized_image.view(batch_size, -1) * (1 - prob_off)
    importance_weight_off = torch.ones((batch_size, 1)) * prob_off

    return torch.cat((importance_weights, importance_weight_off), 1)

def importance_sampled_galaxy_loss(galaxy_vae, image, image_so_far,
                                    var_so_far,
                                    use_importance_sample = True,
                                    use_baseline = True):

    resid_image = image - image_so_far
    class_weights = galaxy_vae.get_pixel_probs(resid_image, var_so_far)
    assert np.all(np.abs(class_weights.sum(1).cpu().detach().numpy() - 1.0) < 1e-6)

    log_q = torch.log(class_weights)

    # kl term
    kl_a = (class_weights * log_q).sum()

    # get importance sampling weights
    if use_importance_sample:
        attn_offset = galaxy_vae.attn_offset
        prob_off = class_weights.detach()[:, -1].view(-1, 1)
        importance_weights = \
            get_importance_weights(image_so_far, attn_offset, prob_off)
    else:
        importance_weights = None

    f_z = lambda i : galaxy_vae.get_loss_conditional_a(\
                        resid_image, image_so_far, var_so_far, i)[0] + kl_a

    ps_loss = get_importance_sampled_loss(f_z, log_q,
                                importance_weights = importance_weights,
                                use_baseline = use_baseline)

    map_locations = torch.argmax(log_q.detach(), dim = 1)
    map_cond_losses = f_z(map_locations).mean()

    return ps_loss, map_cond_losses
