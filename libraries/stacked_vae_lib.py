import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical
import torch.nn.functional as F

import common_utils
import timeit

from copy import deepcopy

import mnist_vae_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model1VAE(nn.Module):
    def __init__(self, sqrt_latent_dim = 6,
                    slen = 28):
        super(Model1VAE, self).__init__()

        self.sqrt_latent_dim = sqrt_latent_dim
        self.latent_dim = sqrt_latent_dim ** 2
        self.slen = slen

        self.encoder = mnist_vae_lib.MLPEncoder(latent_dim = self.latent_dim,
                                    slen = slen,
                                    n_classes = 0)

        self.decoder = mnist_vae_lib.MLPConditionalDecoder(latent_dim = self.latent_dim,
                                            n_classes = 0,
                                            slen = slen)

    def forward(self, image):
        latent_means, latent_std = self.encoder(image, z = None)

        latent_samples = \
            torch.randn(latent_means.shape).to(device) * latent_std + \
                                                            latent_means

        recon_mean, _ = self.decoder(latent_samples, z = None)

        return recon_mean, latent_means, latent_std, latent_samples

    def loss(self, image):
        recon_mean, latent_means, latent_std, latent_samples = \
            self.forward(image)

        # log like term
        loglik_z = common_utils.get_bernoulli_loglik(recon_mean, image)

        # entropy terms
        kl_q_latent = common_utils.get_kl_q_standard_normal(latent_means, \
                                                            latent_std)

        return torch.mean(-loglik_z + kl_q_latent)

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
            image = data['image'].to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            batch_size = image.size()[0]

            loss = self.loss(image)

            if train:
                (loss * num_images).backward()
                optimizer.step()

            avg_loss += loss.data  * (batch_size / num_images)

        return avg_loss

    def train_vae(self, train_loader,
                        test_loader,
                        outfile = './mnist_model1_vae',
                        n_epoch = 200, print_every = 10, save_every = 20,
                        weight_decay = 1e-6, lr = 0.001,
                        save_final_enc = True):

        # define optimizer
        optimizer = optim.Adam([
                {'params': self.parameters(), 'lr': lr}],
                weight_decay=weight_decay)

        iter_array = []
        train_loss_array = []
        test_loss_array = []

        train_loss = self.eval_vae(train_loader)
        test_loss = self.eval_vae(test_loader)
        print('  * init train recon loss: {:.10g};'.format(train_loss))
        print('  * init test recon loss: {:.10g};'.format(test_loss))

        iter_array.append(0)
        train_loss_array.append(train_loss.detach().cpu().numpy())
        test_loss_array.append(test_loss.detach().cpu().numpy())

        for epoch in range(1, n_epoch + 1):
            start_time = timeit.default_timer()

            loss = self.eval_vae(train_loader,
                                    optimizer = optimizer,
                                    train = True)

            elapsed = timeit.default_timer() - start_time
            print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                        epoch, loss, elapsed))

            if epoch % print_every == 0:
                train_loss = self.eval_vae(train_loader)
                test_loss = self.eval_vae(test_loader)

                print('  * train recon loss: {:.10g};'.format(train_loss))
                print('  * test recon loss: {:.10g};'.format(test_loss))

                iter_array.append(epoch)
                train_loss_array.append(train_loss.detach().cpu().numpy())
                test_loss_array.append(test_loss.detach().cpu().numpy())

            if epoch % save_every == 0:
                outfile_every = outfile + '_enc_epoch' + str(epoch)
                print("writing the encoder parameters to " + outfile_every + '\n')
                torch.save(self.encoder.state_dict(), outfile_every)

                outfile_every = outfile + '_dec_epoch' + str(epoch)
                print("writing the decoder parameters to " + outfile_every + '\n')
                torch.save(self.decoder.state_dict(), outfile_every)

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

class StackedModelVAE(nn.Module):
    def __init__(self, model1_vae,
                    m2_latent_dim = 36,
                    n_classes = 10,
                    use_baseline = False):

        super(StackedModelVAE, self).__init__()

        self.model1_vae = model1_vae
        self.model1_sqrt_latent_dim = model1_vae.sqrt_latent_dim
        self.model1_latent_dim = model1_vae.latent_dim
        self.slen = model1_vae.slen

        self.n_classes = n_classes

        # this is just a weird quirk for how the model2 vae got setup ...

        self.model2_vae = mnist_vae_lib.HandwritingVAE(latent_dim = m2_latent_dim,
                        n_classes = n_classes,
                        slen = self.model1_vae.sqrt_latent_dim,
                        use_baseline = use_baseline,
                        normal_likelihood = True)


    def classifier(self, image):
        latent1_mean, latent1_std, latent1_samples = \
                self.model1_encoder_forward(image)

        latent1_mean = latent1_mean.view(-1, self.model1_sqrt_latent_dim,\
                                                self.model1_sqrt_latent_dim)

        class_weights = self.model2_vae.classifier(latent1_mean)

        return class_weights

    def model1_encoder_forward(self, image):
        # pass through model 1 encoder
        latent1_mean, latent1_std = self.model1_vae.encoder(image, z = None)

        # sample
        latent1_samples = \
            torch.randn(latent1_mean.shape).to(device) * latent1_std + \
                                                            latent1_mean

        return latent1_mean, latent1_std, \
                    latent1_samples


    def loss(self, image, true_class_labels = None,
                    num_reinforced = 0):

        latent1_mean, latent1_std, latent1_samples = \
            self.model1_encoder_forward(image)

        latent1_samples = \
                latent1_samples.view(-1, self.model1_sqrt_latent_dim,\
                                            self.model1_sqrt_latent_dim)


        loss, computed_class_weights, ps_loss = self.model2_vae.loss(latent1_samples,
                        true_class_labels = true_class_labels,
                        num_reinforced = num_reinforced)

        return loss, computed_class_weights, ps_loss

    def get_semisupervised_loss(self, unlabeled_images, num_unlabeled_total,
                                    labeled_images = None, labels = None,
                                    alpha = 1.0, num_reinforced = 0):

        latent1_mean_unlab, latent1_std_unlab, latent1_samples_unlab = \
            self.model1_encoder_forward(unlabeled_images)
        latent1_samples_unlab = \
            latent1_samples_unlab.view(-1, self.model1_sqrt_latent_dim,\
                                            self.model1_sqrt_latent_dim)

        if labeled_images is not None:
            latent1_mean_lab, latent1_std_lab, latent1_samples_lab = \
                self.model1_encoder_forward(labeled_images)
            latent1_samples_lab = \
                latent1_samples_lab.view(-1, self.model1_sqrt_latent_dim,\
                                                self.model1_sqrt_latent_dim)
        else:
            latent1_mean_lab = None
            latent1_std_lab = None
            latent1_samples_lab = None

        return self.model2_vae.get_semisupervised_loss(latent1_samples_unlab,
                                        num_unlabeled_total,
                                        labeled_images = latent1_samples_lab,
                                        labels = labels,
                                        alpha = alpha,
                                        num_reinforced = num_reinforced)

    def save_vae(self, outfile, tag):
        outfile_every = outfile + '_enc' + tag
        print("writing the encoder parameters to " + outfile_every + '\n')
        torch.save(self.model2_vae.encoder.state_dict(), outfile_every)

        outfile_every = outfile + '_dec' + tag
        print("writing the decoder parameters to " + outfile_every + '\n')
        torch.save(self.model2_vae.decoder.state_dict(), outfile_every)

        outfile_every = outfile + '_classifier' + tag
        print("writing the classifier parameters to " + outfile_every + '\n')
        torch.save(self.model2_vae.classifier.state_dict(), outfile_every)
