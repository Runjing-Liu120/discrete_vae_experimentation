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

get_linf_diff = lambda x, y : torch.max(torch.abs(x - y))

def assert_diff(x, y, tol = 1e-12):
    assert get_linf_diff(x, y) < tol, \
            'diff = {}'.format(get_linf_diff(x, y))

def get_importance_sampled_loss(f_z, log_q,
                                importance_weights = None,
                                use_baseline = True):
    # class weights from the variational distribution
    assert (log_q.detach() < 0).all()
    class_weights = torch.exp(log_q.detach())

    # why is the tolerance so bad?
    assert_diff(class_weights.sum(1), torch.Tensor([1.0]).to(device), tol = 1e-4)

    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])

    # sample from conditional distribution
    if importance_weights is not None:
        assert importance_weights.shape[0] == log_q.shape[0]
        assert importance_weights.shape[1] == log_q.shape[1]
        # why is the tolerance so bad?
        assert_diff(importance_weights.sum(1), torch.Tensor([1.0]).to(device), tol = 1e-4)

        # sample from importance weights
        z_sample = common_utils.sample_class_weights(importance_weights)

        # reweight accordingly
        importance_weighting = class_weights[seq_tensor, z_sample] / \
                                    importance_weights[seq_tensor, z_sample]
        assert len(importance_weighting) == len(z_sample)
    else:
        z_sample = common_utils.sample_class_weights(class_weights)
        importance_weighting = 1.0

    f_z_i_sample = f_z(z_sample)
    assert len(f_z_i_sample) == len(z_sample)
    log_q_i_sample = log_q[seq_tensor, z_sample]

    if use_baseline:
        z_sample2 = common_utils.sample_class_weights(class_weights)
        baseline = f_z(z_sample2).detach()
    else:
        baseline = 0.0

    reinforce_grad_sample = \
        common_utils.get_reinforce_grad_sample(f_z_i_sample, log_q_i_sample, \
                                                baseline) + f_z_i_sample
    assert len(reinforce_grad_sample) == len(z_sample)

    return (reinforce_grad_sample * importance_weighting).sum()



#####################
# functions for importance sampling galaxy images
def crop_and_normalize_image(image_batch, attn_offset):
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
    normalized_image = crop_and_normalize_image(image_batch, attn_offset)

    importance_weights = normalized_image.view(batch_size, -1) * (1 - prob_off)
    importance_weight_off = torch.ones((batch_size, 1)).to(device) * prob_off

    return torch.cat((importance_weights, importance_weight_off), 1)

def importance_sampled_galaxy_loss(galaxy_vae, image, image_so_far,
                                    var_so_far,
                                    use_importance_sample = True,
                                    use_baseline = True):

    resid_image = image - image_so_far
    class_weights = galaxy_vae.get_pixel_probs(resid_image, var_so_far)
    assert_diff(class_weights.sum(1), torch.Tensor([1.0]).to(device), tol = 1e-4)

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

### This is copied over from galaxy experiments lib
### probably a better way to wrap this to not replicate code
def train_epoch(vae, loader,
                n_samples = 1,
                use_baseline = True,
                use_importance_sample = True,
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

        pm_loss, loss = importance_sampled_galaxy_loss(vae, image = image,
                                            image_so_far = background, # NOTE: we are only doing one detection atm
                                            var_so_far = background,
                                            use_importance_sample = use_importance_sample,
                                            use_baseline = use_baseline)

        if train:
            pm_loss.backward()
            optimizer.step()

        avg_loss += loss * image.shape[0]

    avg_loss /= len(loader.sampler)

    return avg_loss

def train_module(vae, train_loader, test_loader, epochs,
                        use_baseline = True,
                        use_importance_sample = True,
                        lr = 1e-4, weight_decay = 1e-6,
                        save_every = 10,
                        filename = './galaxy_vae_params',
                        seed = 245345):

    optimizer = optim.Adam(
    [{'params': vae.one_galaxy_vae.enc.parameters()},
    {'params': vae.one_galaxy_vae.dec.parameters()},
    {'params': vae.one_galaxy_vae.attn_enc.parameters(), 'lr': 1e-5}],
    lr=lr, weight_decay=weight_decay)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    test_losses_array = []
    batch_losses_array = np.zeros(epochs)
    for epoch in range(0, epochs):
        np.random.seed(seed + epoch)
        start_time = timeit.default_timer()
        batch_loss = train_epoch(vae, train_loader,
                                use_baseline = use_baseline,
                                use_importance_sample = use_importance_sample,
                                train = True,
                                optimizer = optimizer)

        elapsed = timeit.default_timer() - start_time
        print('[{}] loss: {:.0f}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))
        batch_losses_array[epoch] = batch_loss.detach().cpu().numpy()
        np.save(filename + '_batch_losses_array', batch_losses_array[:epoch])

        if epoch % save_every == 0:
            # plot_reconstruction(vae, ds, epoch)
            test_loss = train_epoch(vae, test_loader,
                                    use_baseline = False,
                                    use_importance_sample = False,
                                    train = False,
                                    optimizer = None)

            print('  * test loss: {:.0f}'.format(test_loss))

            save_filename = filename + "_epoch{}.dat".format(epoch)
            print("writing the network's parameters to " + save_filename)
            torch.save(vae.state_dict(), save_filename)

            test_losses_array.append(test_loss.detach())

            np.save(filename + '_test_losses_array', test_losses_array)
