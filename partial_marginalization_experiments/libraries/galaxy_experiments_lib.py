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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
has_cuda = True

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

    def get_pixel_probs(self, resid_image, var_so_far):
        pixel_probs = self.one_galaxy_vae.attn_enc(resid_image, var_so_far)

        # just for myself to make sure I understand this right
        assert (pixel_probs.size(1) - 1) == self.n_discrete_latent

        return pixel_probs

    def sample_pixel(self, pixel_probs):
        pixel_dist = Categorical(pixel_probs)
        return pixel_dist.sample()

    def get_loss_conditional_a(self, resid_image, image_so_far, var_so_far, pixel_1d):
        image = image_so_far + resid_image
        
        is_on = (pixel_1d < (self.n_discrete_latent - 1)).float()

        # pass through galaxy encoder
        pixel_2d = self.one_galaxy_vae.pixel_1d_to_2d(pixel_1d)
        z_mean, z_var = self.one_galaxy_vae.enc(resid_image, pixel_2d)

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

        # get recon loss:
        # NOTE: we will have to the recon means once we do more detections
        recon_means = recon_mean + image_so_far
        recon_vars = recon_var + var_so_far
        recon_losses = -Normal(recon_means, recon_vars.sqrt()).log_prob(image)

        recon_losses = recon_losses.view(image.size(0), -1).sum(1)

        conditional_loss = recon_losses + kl_z

        return conditional_loss, recon_means, recon_vars

    def get_pm_loss(self, image, image_so_far, var_so_far, alpha, topk, use_baseline):

        resid_image = image - image_so_far
        log_q = self.get_pixel_probs(resid_image, var_so_far)

        # kl term
        kl_a = (torch.exp(log_q) * log_q).sum()

        f_z = lambda i : self.get_loss_conditional_a(resid_image, image_so_far, var_so_far, i)[0] + kl_a

        pm_loss = pm_lib.get_partial_marginal_loss(f_z, log_q, alpha, topk,
                                    use_baseline = use_baseline)

        map_locations = torch.argmax(log_q.detach(), dim = 1)
        map_cond_losses = f_z(map_locations).mean()

        return pm_loss, map_cond_losses


def train_epoch(vae, loader,
                alpha = 0.0,
                topk = 0,
                use_baseline = True,
                train = False,
                optimizer = None):
    if train:
        assert optimizer is not None
        vae.train()
    else:
        vae.eval()

    avg_loss = 0.0

    for batch_idx, data in enumerate(loader):
        image = data["image"].to(device)
        background = data["background"].to(device)

        if train:
            optimizer.zero_grad()

        pm_loss, loss = vae.get_pm_loss(image = image,
                                        image_so_far = background,
                                        var_so_far = background, # since  we are only doing 1 detection atm
                                        alpha = alpha,
                                        topk = topk,
                                        use_baseline = use_baseline)
        if train:
            pm_loss.backward()
            optimizer.step()

        avg_loss += loss * image.shape[0]

    avg_loss /= len(loader.sampler)

    return avg_loss

def train_module(vae, train_loader, test_loader, epochs,
                        alpha = 0.0, topk = 0, use_baseline = True,
                        lr = 1e-4, weight_decay = 1e-6,
                        save_every = 10,
                        filename = './galaxy_vae_params',
                        seed = 245345):

    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(0, epochs):
        np.random.seed(seed + epoch)
        start_time = timeit.default_timer()
        batch_loss = train_epoch(vae, train_loader,
                                        alpha = alpha,
                                        topk = topk,
                                        use_baseline = use_baseline,
                                        train = True,
                                        optimizer = optimizer)

        elapsed = timeit.default_timer() - start_time
        print('[{}] loss: {:.0f}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

        if epoch % save_every == 0:
            # plot_reconstruction(vae, ds, epoch)
            test_loss = train_epoch(vae, test_loader,
                                            alpha = alpha,
                                            topk = topk,
                                            use_baseline = False,
                                            train = False)

            print('  * test loss: {:.0f}'.format(test_loss))

            save_filename = filename + "_epoch{}.dat".format(epoch)
            print("writing the network's parameters to " + save_filename)
            torch.save(vae.state_dict(), save_filename)
