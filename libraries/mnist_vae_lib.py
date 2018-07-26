import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal
import torch.nn.functional as F

import common_utils

import timeit

dtype = torch.cuda.float if torch.cuda.is_available() else torch.float

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
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim
        # ndx3 = 2 * self.latent_dim + self.n_classes

        latent_means = h[:, 0:indx1]
        latent_std = torch.exp(h[:, indx1:indx2])
        free_class_weights = h[:, indx2:]
        class_weights = common_utils.get_symplex_from_reals(free_class_weights)

        return latent_means, latent_std, class_weights


class MLPConditionalDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                        n_classes = 10,
                        slen = 28):

        # This takes the latent parameters and returns the
        # mean and variance for the image reconstruction

        super(MLPConditionalDecoder, self).__init__()

        # image/model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        self.fc1 = nn.Linear(latent_dim + n_classes, self.n_pixels)
        self.fc2 = nn.Linear(self.n_pixels, 500)
        self.fc3 = nn.Linear(500, self.n_pixels * 2)


    def forward(self, latent_params, z):
        assert latent_params.shape[1] == self.latent_dim
        assert z.shape[1] == self.n_classes # z should be one hot encoded
        assert latent_params.shape[0] == z.shape[0]

        h = torch.cat((latent_params, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        h = h.view(-1, 2, self.slen, self.slen)

        image_mean = h[:, 0, :, :]
        image_std = torch.exp(h[:, 1, :, :])

        return image_mean, image_std

class HandwritingVAE(nn.Module):

    def __init__(self, latent_dim = 5,
                    n_classes = 10,
                    slen = 28):

        super(HandwritingVAE, self).__init__()

        self.encoder = MLPEncoder(latent_dim = latent_dim,
                                    n_classes = n_classes,
                                    slen = slen)

        self.decoder = MLPConditionalDecoder(latent_dim = latent_dim,
                                                n_classes = n_classes,
                                                slen = slen)

    def encoder_forward(self, image):
        latent_means, latent_std, class_weights = self.encoder(image)

        latent_samples = torch.randn(latent_means.shape, dtype = dtype) * latent_std + latent_means

        return latent_means, latent_std, latent_samples, class_weights

    def decoder_forward(self, latent_samples, z):
        # z should be a vector of integers of length batch_size
        assert len(z) == latent_samples.shape[0]

        one_hot_z = \
            common_utils.get_one_hot_encoding_from_int(z, self.encoder.n_classes)

        image_mean, image_std = self.decoder(latent_samples, one_hot_z)

        return image_mean, image_std

    def loss(self, image):

        latent_means, latent_std, latent_samples, class_weights = \
            self.encoder_forward(image)

        # likelihood term
        loss = 0.0
        for z in range(self.encoder.n_classes):
            batch_z = torch.ones(image.shape[0], dtype = dtype) * z
            image_mu, image_std = self.decoder_forward(latent_samples, batch_z)

            normal_loglik_z = common_utils.get_normal_loglik(image, image_mu,
                                                    image_std, scale = False)
            if not(np.all(np.isfinite(normal_loglik_z.detach().cpu().numpy()))):
                print(z)
                print(image_std)
                assert np.all(np.isfinite(normal_loglik_z.detach().cpu().numpy()))

            loss = - (class_weights[:, z] * normal_loglik_z).sum()

        # kl term for latent parameters
        # (assuming standard normal prior)
        kl_q_latent = common_utils.get_kl_q_standard_normal(latent_means, \
                                                            latent_std)
        assert np.isfinite(kl_q_latent.detach().cpu().numpy())

        # entropy term for class weights
        # (assuming uniform prior)
        kl_q_z = -common_utils.get_multinomial_entropy(class_weights)
        assert np.isfinite(kl_q_z.detach().cpu().numpy())

        loss += (kl_q_latent + kl_q_z)

        return loss / image.size()[0]

    def eval_vae(self, train_loader, optimizer = None, train = False):
        if train:
            self.train()
            assert optimizer is not None
        else:
            self.eval()

        avg_loss = 0.0

        num_images = train_loader.dataset.__len__()
        i = 0

        for batch_idx, data in enumerate(train_loader):
            # first entry of data is the actual image
            # the second entry is the true class label
            if torch.cuda.is_available():
                image = data[0].cuda()
            else:
                image = data[0]

            i+=1; print('batch {}'.format(i))

            if optimizer is not None:
                optimizer.zero_grad()

            batch_size = image.size()[0]

            loss = self.loss(image)  * (batch_size / num_images)

            if train:
                loss.backward()
                optimizer.step()

            avg_loss += loss.data

        return avg_loss

    def train_module(self, train_loader, test_loader,
                    set_true_lens_params = False,
                    set_true_lens_on = False,
                    set_true_source = False,
                    outfile = './mnist_vae',
                    n_epoch = 200, print_every = 10, save_every = 20,
                    weight_decay = 1e-6, lr = 0.001,
                    save_final_enc = True):

        optimizer = optim.Adam(self.parameters(), lr=lr,
                                weight_decay=weight_decay)

        iter_array = []
        train_loss_array = []
        test_loss_array = []

        train_loss = self.eval_vae(train_loader)
        test_loss = self.eval_vae(test_loader)
        print('  * init train recon loss: {:.10g};'.format(train_loss))
        print('  * init test recon loss: {:.10g};'.format(test_loss))

        iter_array.append(0)
        train_loss_array.append(train_loss.detach().numpy())
        test_loss_array.append(test_loss.detach().numpy())

        for epoch in range(1, n_epoch + 1):
            start_time = timeit.default_timer()

            batch_loss = self.eval_vae(train_loader,
                                        optimizer = optimizer,
                                        train = True)

            elapsed = timeit.default_timer() - start_time
            print('[{}] loss: {:.10g}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

            if epoch % print_every == 0:
                train_loss = self.eval_vae(train_loader)
                test_loss = self.eval_vae(test_loader)

                print('  * train recon loss: {:.10g};'.format(train_loss))
                print('  * test recon loss: {:.10g};'.format(test_loss))

                iter_array.append(epoch)
                train_loss_array.append(train_loss.detach().numpy())
                test_loss_array.append(test_loss.detach().numpy())

            if epoch % save_every == 0:
                outfile_every = outfile + '_epoch' + str(epoch)
                print("writing the encoder parameters to " + outfile_every + '\n')
                torch.save(self.parameters(), outfile_every)

        if save_final_enc:
            outfile_final = outfile + '_enc_final'
            print("writing the encoder parameters to " + outfile_final + '\n')
            torch.save(self.encoder.state_dict(), outfile_final)

            outfile_final = outfile + '_dec_final'
            print("writing the decoder parameters to " + outfile_final + '\n')
            torch.save(self.decoder.state_dict(), outfile_final)

            loss_array = np.zeros((3, len(iter_array)))
            loss_array[0, :] = iter_array
            loss_array[1, :] = train_loss_array
            loss_array[2, :] = test_loss_array
            np.savetxt(outfile + 'loss_array.txt', loss_array)
