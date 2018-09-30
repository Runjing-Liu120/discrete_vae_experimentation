import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import timeit

from copy import deepcopy

import sys
sys.path.insert(0, '../../../celeste_net/')

from celeste_net import PixelAttention, OneGalaxyEncoder, OneGalaxyDecoder
from datasets import Synthetic

from torch.utils.data import DataLoader, sampler

import partial_marginalization_lib as pm_lib

has_cuda = False

def get_train_test_data(ds, batch_size):

    tt_split = int(0.1 * len(ds))
    test_indices = np.mgrid[:tt_split]
    train_indices = np.mgrid[tt_split:len(ds)]

    test_loader = DataLoader(ds, batch_size=batch_size,
                             num_workers=2, pin_memory=has_cuda,
                             sampler=sampler.SubsetRandomSampler(test_indices))
    train_loader = DataLoader(ds, batch_size=batch_size,
                              num_workers=2, pin_memory=has_cuda,
                              sampler=sampler.SubsetRandomSampler(train_indices))

    return train_loader, test_loader

class CelesteRNN(nn.Module):

    def __init__(self, sout, one_galaxy_vae, max_detections=1):
        super(CelesteRNN, self).__init__()

        self.one_galaxy_vae = one_galaxy_vae
        self.max_detections = max_detections

        # number of discrete random variables
        # TODO: please check this ...
        self.n_discrete_latent = (sout - 2 * self.one_galaxy_vae.attn_enc.attn_offset)**2

    def get_pixel_probs(self, image, image_var):
        pixel_probs = self.one_galaxy_vae.attn_enc(image, image_var)

        # just for myself to make sure I understand this right
        assert (pixel_probs.size(1) - 1) == self.n_discrete_latent

        return pixel_probs

    def sample_pixel(self, pixel_probs):
        pixel_dist = Categorical(pixel_probs)
        return pixel_dist.sample()

    def get_loss_conditional_a(self, image, image_var, pixel_1d):

        is_on = (pixel_1d < (self.n_discrete_latent - 1)).float()

        # pass through galaxy encoder
        pixel_2d = self.one_galaxy_vae.pixel_1d_to_2d(pixel_1d)
        z_mean, z_var = self.one_galaxy_vae.enc(image, pixel_2d)

        # sample z
        q_z = Normal(z_mean, z_var.sqrt())
        z_sample = q_z.rsample()

        # kl term for continuous latent vars
        log_q_z = q_z.log_prob(z_sample).sum(1)
        p_z = Normal(torch.zeros_like(z_sample), torch.ones_like(z_sample))
        log_p_z = p_z.log_prob(z_sample).sum(1)
        kl_z = is_on * (log_q_z - log_p_z)

        # run through decoder
        recon_mean, recon_var = self.one_galaxy_vae.dec(pixel_2d, is_on, z_sample)

        # get recon loss
        recon_losses = -Normal(recon_mean, (recon_var + image_var).sqrt()).log_prob(image)
        recon_losses = recon_losses.view(image.size(0), -1).sum(1)

        conditional_loss = recon_losses + kl_z

        return conditional_loss, recon_mean, recon_var

    def get_pm_loss(self, image, image_var, alpha, topk, use_baseline):
        log_q = self.get_pixel_probs(image, image_var)

        # kl term
        kl_a = (torch.exp(log_q) * log_q).sum()

        f_z = lambda i : self.get_loss_conditional_a(image, image_var, i)[0] + kl_a

        pm_loss = pm_lib.get_partial_marginal_loss(f_z, log_q, alpha, topk,
                                    use_baseline = use_baseline)

        map_locations = torch.argmax(log_q.detach(), dim = 1)
        map_cond_losses = f_z(map_locations).mean()

        return pm_loss, map_cond_losses
