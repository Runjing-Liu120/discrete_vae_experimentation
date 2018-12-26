import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical

import modeling_lib
import mnist_data_utils

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

    def forward(self, latent_samples):

        h = self.tanh(self.fc1(latent_samples))
        h = self.fc2(h)

        return self.sigmoid(h).view(-1, 1, self.slen, self.slen)

class HandwritingVAE(nn.Module):
    def __init__(self, latent_dim = 5,
                        slen = 28):

        super(HandwritingVAE, self).__init__()

        self.latent_dim = latent_dim
        self.slen = slen

        self.encoder = MLPEncoder(self.latent_dim, self.slen)
        self.decoder = MLPDecoder(self.latent_dim, self.slen)

    def forward(self, image):

        # get latent means and std
        latent_mean, latent_log_std = self.encoder(image)

        # sample latent params
        latent_samples = torch.randn(latent_mean.shape).to(device) * \
                            torch.exp(latent_log_std) + latent_mean

        # pass through decoder
        recon_mean = self.decoder(latent_samples)

        return recon_mean, latent_mean, latent_log_std, latent_samples

    def get_loss(self, image):

        recon_mean, latent_mean, latent_log_std, latent_samples = \
            self.forward(image)

        # kl term
        kl_q = modeling_lib.get_kl_q_standard_normal(latent_mean, latent_log_std)

        # bernoulli likelihood
        loglik = modeling_lib.get_bernoulli_loglik(recon_mean, image)

        return -loglik + kl_q


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class PixelAttention(nn.Module):
    def __init__(self, slen):

        super(PixelAttention, self).__init__()

        # attention mechanism

        # convolution layers
        self.attn = nn.Sequential(
            nn.Conv2d(1, 7, 3, padding=0),
            nn.ReLU(),

            nn.Conv2d(7, 7, 3, padding=0),
            nn.ReLU(),

            nn.Conv2d(7, 7, 3, padding=0),
            nn.ReLU(),

            nn.Conv2d(7, 1, 3, padding=0),
            Flatten())

        # one more fully connected layer
        self.slen = slen
        self.fc1 = nn.Linear((self.slen - 8)**2, self.slen**2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        h = self.attn(image)
        h = self.fc1(h)

        return self.softmax(h)

class MovingHandwritingVAE(nn.Module):
    def __init__(self, latent_dim = 5,
                        mnist_slen = 28,
                        full_slen = 68):

        super(MovingHandwritingVAE, self).__init__()

        # mnist_slen is size of the mnist digit
        # full_slen is the size of the padded image

        self.latent_dim = latent_dim
        self.mnist_slen = mnist_slen
        self.full_slen = full_slen

        self.mnist_vae = HandwritingVAE(latent_dim = self.latent_dim,
                                        slen = self.mnist_slen + 1)

        self.pixel_attention = PixelAttention(slen = self.full_slen)

        # cache meshgrid required for padding images
        r0 = (self.full_slen - 1) / 2
        self.grid_out = \
            torch.FloatTensor(
                np.mgrid[0:self.full_slen, 0:self.full_slen].transpose() - r0)

        # cache meshgrid required for cropping image
        r = self.mnist_slen // 2
        self.grid0 = torch.from_numpy(\
                    np.mgrid[(-r):(r+1), (-r):(r+1)].transpose([2, 1, 0]))

    def forward(self, image):
        # image should be N x slen x slen
        assert len(image.shape) == 4
        assert image.shape[1] == 1
        assert image.shape[2] == self.full_slen
        assert image.shape[3] == self.full_slen

        # the pixel where to attend.
        # the CNN requires a 4D input.
        pixel_probs = self.pixel_attention(image)

        # sample pixel
        categorical = Categorical(pixel_probs)
        pixel_1d_sample = categorical.sample().detach()

        # crop image about sampled pixel
        pixel_2d_sample = mnist_data_utils.pixel_1d_to_2d(self.full_slen,
                                    padding = 0,
                                    pixel_1d = pixel_1d_sample)

        image_cropped = mnist_data_utils.crop_image(image,
                            pixel_2d_sample, grid0 = self.grid0)

        # pass through mnist vae
        recon_mean_cropped, latent_mean, latent_log_std, latent_samples = \
            self.mnist_vae(image_cropped)

        # re-pad image
        recon_mean = \
            mnist_data_utils.pad_image(recon_mean_cropped,
                                        pixel_2d_sample,
                                        grid_out = self.grid_out)

        return recon_mean, latent_mean, latent_log_std, latent_samples, \
                    pixel_probs, pixel_1d_sample

    def get_loss(self, image):

        # forward
        recon_mean, latent_mean, latent_log_std, latent_samples, \
                    pixel_probs, pixel_1d_sample = \
                        self.forward(image)

        # kl term
        kl_latent = \
            modeling_lib.get_kl_q_standard_normal(latent_mean, latent_log_std)
        kl_pixel_probs = \
            modeling_lib.get_multinomial_kl(pixel_probs)

        # bernoulli likelihood
        loglik = modeling_lib.get_bernoulli_loglik(recon_mean, image)

        return -loglik + kl_latent + kl_pixel_probs
