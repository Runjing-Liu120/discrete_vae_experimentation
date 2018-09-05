import torch

from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################
# Some terms for the ELBO
#############################
def get_normal_loglik(x, mean, std, scale = False):
    recon_losses = \
        Normal(mean, std + 1e-8).log_prob(x)

    if scale:
        factor = torch.prod(torch.Tensor([x.size()])) / x.size()[0]
    else:
        factor = 1.0

    return (recon_losses / factor).view(x.size(0), -1).sum(1)

def get_bernoulli_loglik(pi, x):
    loglik = (x * torch.log(pi + 1e-8) + (1 - x) * torch.log(1 - pi + 1e-8))
    return loglik.view(x.size(0), -1).sum(1)

softmax = torch.nn.Softmax(dim=1)

def get_kl_q_standard_normal(mu, sigma):
    # The KL between a Gaussian variational distribution
    # and a standard normal
    mu = mu.view(mu.shape[0], -1)
    sigma = sigma.view(sigma.shape[0], -1)
    return 0.5 * torch.sum(-1 - torch.log(sigma**2 + 1e-8) + \
                                mu**2 + sigma**2, dim = 1)

def get_multinomial_entropy(z):
    return (- z * torch.log(z + 1e-8)).sum()

############################
# Other miscillaneous utils
#############################
def get_symplex_from_reals(unconstrained_mat):
    # returns a vector on the symplex from the unconstrained
    # real parametrization

    # first column is reference value
    aug_unconstrained_mat = torch.cat([
                            torch.zeros((unconstrained_mat.shape[0], 1)).to(device),
                            unconstrained_mat], 1)

    return softmax(aug_unconstrained_mat)

def get_one_hot_encoding_from_int(z, n_classes):
    # z is a sequence of integers in {0, ...., n_classes}
    #  corresponding to categorires
    # we return a matrix of shape len(z) x n_classes
    # corresponding to the one hot encoding of z

    assert (torch.max(z) + 1) <= n_classes

    batch_size = len(z)
    one_hot_z = torch.zeros((batch_size, n_classes)).to(device)

    for i in range(n_classes):
        one_hot_z[z == i, i] = 1.

    return one_hot_z

#####################
# Some functions to examine VAE results
def get_classification_accuracy(loader, classifier,
                                    return_wrong_images = False,
                                    max_images = 1000):

    n_images = 0.0
    accuracy = 0.0

    slen = loader.sampler.data_source[0]['image'].shape[0]
    wrong_images = torch.zeros((0, slen, slen))
    wrong_labels = torch.LongTensor(0)

    for batch_idx, data in enumerate(loader):
        class_weights = classifier(data['image'])

        z_ind = torch.argmax(class_weights, dim = 1)

        accuracy += torch.sum(z_ind == data['label']).float()
        # print(accuracy)

        if return_wrong_images:
            wrong_indx = 1 - (z_ind == data['label'])
            wrong_images = torch.cat((wrong_images,
                                    data['image'][wrong_indx, :, :]),
                                    dim = 0)
            wrong_labels = torch.cat((wrong_labels,
                                data['label'][wrong_indx]))
        else:
            wrong_images = None
            wrong_labels = None

        n_images += len(z_ind)
        if n_images > 1000:
            break

    return accuracy / n_images, wrong_images, wrong_labels

def get_reconstructions(vae, image):
    class_weights = vae.classifier(image)

    z_ind = torch.argmax(class_weights, dim = 1)
    z_ind_one_hot = \
        get_one_hot_encoding_from_int(z_ind, vae.n_classes)

    latent_means, latent_std, latent_samples = \
        vae.encoder_forward(image, z_ind_one_hot)


    image_mu = vae.decoder_forward(latent_means, z_ind_one_hot)

    return image_mu, z_ind
