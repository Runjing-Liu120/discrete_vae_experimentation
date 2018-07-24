import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal
import torch.nn.functional as F

import common_utils

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    n_classes = 10,
                    slen = 28):
        # the encoder returns the mean and variance of the latent parameters
        # and the unconstrained symplex parametrization for the classes

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels, 500)
        self.fc2 = nn.Linear(500, self.n_pixels)
        self.fc3 = nn.Linear(self.n_pixels, (n_classes - 1) + latent_dim * 2)



    def forward(self, image):

        # feed through neural network
        z = image.view(-1, self.n_pixels)

        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim
        indx3 = 2 * self.latent_dim + self.n_classes

        latent_means = z[:, 0:indx1]
        latent_std = torch.exp(z[:, indx1:indx2])
        free_class_weights = z[:, indx2:indx3]

        return latent_means, latent_std, free_class_weights


class MLPConditionalDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                        slen = 28):

        # This takes the latent parameters and returns the
        # mean and variance for the image reconstruction

        super(MLPConditionalDecoder, self).__init__()

        # image/model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen

        self.fc1 = nn.Linear(latent_dim, self.n_pixels)
        self.fc2 = nn.Linear(self.n_pixels, 500)
        self.fc3 = nn.Linear(500, self.n_pixels * 2)


    def forward(self, latent_params):
        latent_params = latent_params.view(-1, self.latent_dim)

        z = F.relu(self.fc1(latent_params))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)

        z = z.view(-1, 2, self.slen, self.slen)

        image_mean = z[:, 0, :, :]
        image_std = torch.exp(z[:, 1, :, :])

        return image_mean, image_std

class HandwritingVAE(nn.Module):

    def __init__(self, latent_dim = 5,
                    n_classes = 9,
                    slen = 28):

        super(HandwritingVAE, self).__init__()

        self.encoder = MLPEncoder(latent_dim = latent_dim,
                                    n_classes = n_classes,
                                    slen = slen)

        # one decoder for each classes
        self.decoder_list = [
            MLPConditionalDecoder(latent_dim = latent_dim, slen = slen) for
            k in range(n_classes)
        ]

    def encoder_forward(self, image):
        latent_means, latent_std, free_class_weights = self.encoder(image)

        class_weights = common_utils.get_symplex_from_reals(free_class_weights)

        latent_samples = torch.randn(latent_means.shape) * latent_std + latent_means

        return latent_means, latent_std, latent_samples, class_weights

    def decoder_forward(self, latent_samples, z):
        assert z <= len(self.decoder_list)

        image_mean, image_std = self.decoder_list[z](latent_samples)

        return image_mean, image_std

    def loss(self, image):

        latent_means, latent_std, latent_samples, class_weights = \
            self.encoder_forward(image)

        # likelihood term
        loss = 0.0
        for z in range(self.encoder.n_classes):
            image_mu, image_std = self.decoder_forward(latent_samples, z)

            normal_loglik_z = common_utils.get_normal_loglik(image, image_mu,
                                                    image_std, scale = False)

            loss = - (class_weights[:, z] * normal_loglik_z).sum()

        # kl term for latent parameters
        # (assuming standard normal prior)
        kl_q_latent = common_utils.get_kl_q_standard_normal(latent_means, \
                                                            latent_std).sum()

        # entropy term for class weights
        # (assuming uniform prior)
        kl_q_z = common_utils.get_multinomial_entropy(class_weights).sum()

        loss -= (kl_q_latent + kl_q_z)

        return loss / image.size()[0]

    def eval_vae(self, train_loader, optimizer = None, train = False):
        if train:
            self.train()
            assert optimizer is not None
        else:
            self.eval()

        avg_loss = 0.0

        num_images = train_loader.dataset.__len__()

        for batch_idx, data in enumerate(train_loader):
            image = data[0] # first entry in the tuple is the actual image
            # the second entry is the true class label

            if optimizer is not None:
                optimizer.zero_grad()

            batch_size = image.size()[0]

            loss = self.loss(image)

            # do we need to scale this for the gradient to be unbiased?
            if train:
                loss.backward()
                optimizer.step()

            avg_loss += loss.data * (batch_size / num_images)

        return avg_loss

    def train_module(self, train_loader, test_loader,
                    set_true_lens_params = False,
                    set_true_lens_on = False,
                    set_true_source = False,
                    outfile = './enc',
                    n_epoch = 200, print_every = 10, save_every = 20,
                    weight_decay = 1e-6, lr = 0.001,
                    save_final_enc = True):

        optimizer = optim.Adam(self.parameters(), lr=lr,
                                weight_decay=weight_decay)

        train_loss = self.eval_vae(train_loader)
        test_loss = self.eval_vae(test_loader)
        print('  * init train recon loss: {:.10g};'.format(train_loss))
        print('  * init test recon loss: {:.10g};'.format(test_loss))

        for epoch in range(1, n_epoch + 1):
            start_time = timeit.default_timer()

            batch_loss = self.eval_vae(train_loader,
                                        optimizer = optimizer,
                                        train = True)

            elapsed = timeit.default_timer() - start_time
            print('[{}] loss: {:.0f}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

            if epoch % print_every == 0:
                train_loss = self.eval_vae(train_loader)
                test_loss = self.eval_vae(test_loader)

                print('  * train recon loss: {:.10g};'.format(train_loss))
                print('  * test recon loss: {:.10g};'.format(test_loss))

            if epoch % save_every == 0:
                outfile_every = outfile + '_epoch' + str(epoch)
                print("writing the encoder parameters to " + outfile_every + '\n')
                torch.save(self.parameters(), outfile_every)

        if save_final_enc:
            outfile_final = outfile + '_final'
            print("writing the encoder parameters to " + outfile_final + '\n')
            torch.save(self.lensing_vae.enc.state_dict(), outfile_final)
