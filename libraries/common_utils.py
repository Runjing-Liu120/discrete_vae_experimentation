import torch

from torch.distributions import Normal

def get_normal_loglik(x, mean, std, scale = False):
    recon_losses = \
        Normal(mean, std).log_prob(x)

    if scale:
        factor = torch.prod(torch.Tensor([x.size()]))
    else:
        factor = 1.0

    return (recon_losses / factor).view(x.size(0), -1).sum(1)


softmax = torch.nn.Softmax(dim=1)

def get_kl_q_standard_normal(mu, sigma):
    # The KL between a Gaussian variational distribution
    # and a standard normal
    return - 0.5 * torch.sum(-1 - torch.log(sigma**2) + \
                                mu**2 + sigma**2, dim = 1)

def get_multinomial_entropy(z):
    return (- z * torch.log(z)).sum(-1)

def get_symplex_from_reals(unconstrained_mat):
    # returns a vector on the symplex from the unconstrained
    # real parametrization

    # first column is reference value
    aug_unconstrained_mat = torch.cat([
                            torch.zeros((unconstrained_mat.shape[0], 1)),
                            unconstrained_mat], 1)

    return softmax(aug_unconstrained_mat)
