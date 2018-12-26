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

    def forward(self, image):
        # image should be N x slen x slen
        assert len(image.shape) == 3
        assert image.shape[1] == self.full_slen
        assert image.shape[2] == self.full_slen

        # the pixel where to attend.
        # the CNN requires a 4D input.
        image_ = image.view(image.shape[0], -1, image.shape[-1], image.shape[-1])
        pixel_probs = self.pixel_attention(image_)

        # sample pixel
        categorical = Categorical(pixel_probs)
        pixel_1d_sample = categorical.sample().detach()

        # crop image about sampled pixel
        pixel_2d_sample = mnist_data_utils.pixel_1d_to_2d(self.full_slen,
                                    padding = 0,
                                    pixel_1d = pixel_1d_sample)

        image_cropped = mnist_data_utils.crop_image(image,
                            pixel_2d_sample, self.mnist_slen)

        # pass through mnist vae
        recon_mean_cropped, latent_mean, latent_log_std, latent_samples = \
            self.mnist_vae(image_cropped)

        # re-pad image
        recon_mean = \
            mnist_data_utils.pad_image(recon_mean_cropped,
                                        pixel_2d_sample,
                                        self.full_slen)

        return recon_mean, latent_mean, latent_log_std, latent_samples, \
                    pixel_probs, pixel_1d_sample

    def get_loss(self, image, recon_mean, latent_mean, latent_log_std):

        # kl term
        kl_q = modeling_lib.get_kl_q_standard_normal(latent_mean, latent_log_std)

        # bernoulli likelihood
        loglik = modeling_lib.get_bernoulli_loglik(recon_mean, image)

        return -loglik + kl_q


def eval_vae(vae, loader, \
                optimizer = None,
                train = False):
    if train:
        vae.train()
        assert optimizer is not None
    else:
        vae.eval()

    avg_loss = 0.0

    num_images = len(loader.dataset)

    for batch_idx, data in enumerate(loader):

        if optimizer is not None:
            optimizer.zero_grad()

        image = data['image'].to(device)

        loss = vae.get_loss(image).sum()

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss.data  / num_images

    return avg_loss

def train_vae(vae, train_loader, test_loader, optimizer,
                    outfile = './mnist_vae_semisupervised',
                    n_epoch = 200, print_every = 10, save_every = 20):

    # get losses
    train_loss = eval_vae(vae, train_loader, train = False)
    test_loss = eval_vae(vae, test_loader, train = False)

    print('  * init train recon loss: {:.10g};'.format(train_loss))
    print('  * init test recon loss: {:.10g};'.format(test_loss))

    for epoch in range(1, n_epoch + 1):
        start_time = timeit.default_timer()

        loss = eval_vae(vae, train_loader,
                                optimizer = optimizer,
                                train = True)

        elapsed = timeit.default_timer() - start_time
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, loss, elapsed))

        if epoch % print_every == 0:
            train_loss = eval_vae(vae, train_loader, train = False)
            test_loss = eval_vae(vae, test_loader, train = False)

            print('  * train recon loss: {:.10g};'.format(train_loss))
            print('  * test recon loss: {:.10g};'.format(test_loss))

        if epoch % save_every == 0:
            outfile_every = outfile + '_epoch' + str(epoch)
            print("writing the parameters to " + outfile_every + '\n')
            torch.save(vae.state_dict(), outfile_every)

    outfile_final = outfile + '_final'
    print("writing the parameters to " + outfile_final + '\n')
    torch.save(vae.state_dict(), outfile_final)
