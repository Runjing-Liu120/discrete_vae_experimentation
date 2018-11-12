import torch
from torch.autograd import Variable
from torch import nn

import mnist_utils

import cifar_data_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# code adapted from https://github.com/kuc2477/pytorch-vae/blob/master/model.py

class CIFARConditionalVAE(nn.Module):
    def __init__(self, slen, channel_num, kernel_num, z_size, n_classes, use_cifar100 = True):
        # configurations
        super().__init__()
        self.slen = slen # side length of image
        self.channel_num = channel_num # number of channels in an image
        self.kernel_num = kernel_num # "channel number" in the encoded image
        self.z_size = z_size # size of latent dimension

        self.n_classes = n_classes

        self.use_cifar100 = use_cifar100

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = slen // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # encoder then encorporates the class label
        self.cond_encoder = nn.Sequential(
            self._linear(self.feature_volume + self.n_classes, 128),
            self._linear(128, 128),
            self._linear(128, z_size * 2, relu = False),
        )

        # decodr that encorporates class label
        self.cond_decoder = nn.Sequential(
            self._linear(z_size + self.n_classes, 128),
            self._linear(128, 128),
            self._linear(128, self.feature_volume),
        )

        # self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        # self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        # self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num, relu = True),
            nn.Sigmoid()
        )

    def cond_encoder_forward(self, encoded, label):
        # this is my added function, where we take the convolved image and
        # incorporate the label.

        assert encoded.shape[0] == len(label)

        # append one hot encoding of labels
        one_hot_labels = mnist_utils.get_one_hot_encoding_from_int(label, self.n_classes)
        encoded = encoded.view(-1, self.feature_volume)

        h = torch.cat((encoded, one_hot_labels), dim = 1)

        h = self.cond_encoder(h)

        indx1 = self.z_size
        indx2 = 2 * self.z_size

        latent_means = h[:, 0:indx1]
        latent_log_var = h[:, indx1:indx2]

        return latent_means, latent_log_var, one_hot_labels

    def cond_decoder_forward(self, latent_samples, one_hot_labels):
        assert latent_samples.shape[0] == one_hot_labels.shape[0]
        assert latent_samples.shape[1] == self.z_size
        assert one_hot_labels.shape[1] == self.n_classes

        h = torch.cat((latent_samples, one_hot_labels), dim = 1)

        return self.cond_decoder(h).view(-1, self.kernel_num,
                                                self.feature_size,
                                                self.feature_size)

    def forward(self, x, label):

        n_obs = x.shape[0]

        # encode x
        # this is (N x kernel_num x (slen / 8) x (slen / 8))
        if self.use_cifar100:
            x = x * cifar_data_utils.CIFAR100_STD_TENSOR.to(device) + \
                        cifar_data_utils.CIFAR100_MEAN_TENSOR.to(device)

        else:
            x = x * cifar_data_utils.CIFAR10_STD_TENSOR.to(device) + \
                        cifar_data_utils.CIFAR10_MEAN_TENSOR.to(device)

        encoded = self.encoder(x)

        assert encoded.shape[0] == n_obs

        # combine encoded with the label,
        # and pass through some fully connected layers
        # get latent means and variances
        # means and variances are of shape (N x z_size)
        latent_means, latent_log_var, one_hot_labels = \
            self.cond_encoder_forward(encoded, label)
        latent_std = torch.exp(0.5 * latent_log_var)

        # sample
        latent_samples = self.z(latent_means, latent_log_var)

        assert latent_means.shape[0] == n_obs
        assert latent_std.shape[0] == n_obs
        assert latent_samples.shape[0] == n_obs

        # combine latent sample with labels
        # pass through a few fully connected layers
        latent_samples_label = \
            self.cond_decoder_forward(latent_samples, one_hot_labels)

        assert latent_samples_label.shape[0] == n_obs

        # deconvolve and reconstruct x
        image_mean = self.decoder(latent_samples_label)
        image_var = None

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return latent_means, latent_std, latent_samples, image_mean, image_var

    # ==============
    # VAE components
    # ==============

    # def q(self, encoded):
    #     unrolled = encoded.view(-1, self.feature_volume)
    #     return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).to(device)
        )
        return eps.mul(std).add_(mean)

    # def reconstruction_loss(self, x_reconstructed, x):
    #     return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)
    #
    # def kl_divergence_loss(self, mean, logvar):
    #     return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

    # =====
    # Utils
    # =====

    # @property
    # def name(self):
    #     return (
    #         'VAE'
    #         '-{kernel_num}k'
    #         '-{label}'
    #         '-{channel_num}x{slen}x{slen}'
    #     ).format(
    #         label=self.label,
    #         kernel_num=self.kernel_num,
    #         slen=self.slen,
    #         channel_num=self.channel_num,
    #     )

    # def sample(self, size):
    #     z = Variable(
    #         torch.randn(size, self.z_size).to(device)
    #     )
    #     z_projected = self.project(z).view(
    #         -1, self.kernel_num,
    #         self.feature_size,
    #         self.feature_size,
    #     )
    #     return self.decoder(z_projected).data

    # def _is_on_cuda(self):
    #     return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num, relu = True):

        if relu:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    channel_num, kernel_num,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm2d(kernel_num),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    channel_num, kernel_num,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm2d(kernel_num),
            )


    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)
