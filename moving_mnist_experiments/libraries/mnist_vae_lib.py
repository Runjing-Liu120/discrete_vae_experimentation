import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

import modeling_lib

import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28):

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, self.latent_dim * 2)

        self.tanh = torch.nn.Tanh()

    def forward(self, image):

        h = image.view(-1, self.n_pixels)

        h = self.tanh(self.fc1(h))
        h = self.fc2(h)

        latent_mean = h[:, 0:self.latent_dim]
        latent_log_std = h[:, self.latent_dim:(2 * self.latent_dim)]

        return latent_mean, latent_log_std

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28):

        super(MLPDecoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, self.n_pixels)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, latent_params):

        h = self.tanh(self.fc1(latent_params))
        h = self.fc2(h)

        return self.sigmoid(h).view(-1, self.slen, self.slen)

class HandwritingVAE(nn.Module):
    def __init__(self, latent_dim = 5,
                        slen = 28):

        super(HandwritingVAE, self).__init__()

        self.latent_dim = latent_dim
        self.slen = slen

        self.encoder = MLPEncoder(self.latent_dim, self.slen)
        self.decoder = MLPDecoder(self.latent_dim, self.slen)

    def forward(self, image):
        # image should be N x slen x slen
        assert len(image.shape) == 3
        assert image.shape[1] == self.slen
        assert image.shape[2] == self.slen

        # get latent means and std
        latent_mean, latent_log_std = self.encoder(image)

        # sample latent params
        latent_params = torch.randn(latent_mean.shape).to(device) * \
                            torch.exp(latent_log_std) + latent_mean

        # pass through decoder
        recon_mean = self.decoder(latent_params)

        return recon_mean, latent_mean, latent_log_std, latent_params

    def get_loss(self, image, recon_mean, latent_mean, latent_log_std):

        # kl term
        kl_q = modeling_lib.get_kl_q_standard_normal(latent_mean, latent_log_std)

        # bernoulli likelihood
        loglik = modeling_lib.get_bernoulli_loglik(recon_mean, image)

        return -loglik + kl_q

    def eval_vae(self, loader, \
                    optimizer = None,
                    train = False):
        if train:
            self.train()
            assert optimizer is not None
        else:
            self.eval()

        avg_loss = 0.0

        num_images = len(loader.dataset)

        for batch_idx, data in enumerate(loader):

            if optimizer is not None:
                optimizer.zero_grad()

            image = data['image'].to(device)

            recon_mean, latent_mean, latent_log_std, latent_params = \
                self.forward(image)

            loss = self.get_loss(image, recon_mean,
                                latent_mean, latent_log_std).sum()

            if train:
                loss.backward()
                optimizer.step()

            avg_loss += loss.data  / num_images

        return avg_loss

    def train_vae(self, train_loader, test_loader,
                        outfile = './mnist_vae_semisupervised',
                        n_epoch = 200, print_every = 10, save_every = 20,
                        weight_decay = 1e-6, lr = 0.001):

        # define optimizer
        optimizer = optim.Adam([
                    {'params': self.parameters(), 'lr': lr}],
                        weight_decay=weight_decay)

        # get losses
        train_loss = self.eval_vae(train_loader, train = False)
        test_loss = self.eval_vae(test_loader, train = False)

        print('  * init train recon loss: {:.10g};'.format(train_loss))
        print('  * init test recon loss: {:.10g};'.format(test_loss))

        for epoch in range(1, n_epoch + 1):
            start_time = timeit.default_timer()

            loss = self.eval_vae(train_loader,
                                    optimizer = optimizer,
                                    train = True)

            elapsed = timeit.default_timer() - start_time
            print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                        epoch, loss, elapsed))

            if epoch % print_every == 0:
                train_loss = self.eval_vae(train_loader, train = False)
                test_loss = self.eval_vae(test_loader, train = False)

                print('  * train recon loss: {:.10g};'.format(train_loss))
                print('  * test recon loss: {:.10g};'.format(test_loss))

            if epoch % save_every == 0:
                outfile_every = outfile + '_epoch' + str(epoch)
                print("writing the parameters to " + outfile_every + '\n')
                torch.save(self.state_dict(), outfile_every)

        outfile_final = outfile + '_final'
        print("writing the parameters to " + outfile_final + '\n')
        torch.save(self.state_dict(), outfile_final)
