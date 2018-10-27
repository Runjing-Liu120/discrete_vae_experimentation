image_mu, image_std, latent_means, latent_std, latent_samples = \
    self.forward_conditional(image, batch_z)

# likelihood term
normal_loglik_z = common_utils.get_normal_loglik(image, image_mu,
                                        image_std, scale = False)

if not(np.all(np.isfinite(normal_loglik_z.detach().cpu().numpy()))):
    print(z)
    print(image_std)
    assert np.all(np.isfinite(normal_loglik_z.detach().cpu().numpy()))

loss -= (class_weights[:, z] * normal_loglik_z).sum()

# entropy term
kl_q_latent = common_utils.get_kl_q_standard_normal(latent_means, \
                                                    latent_std)
assert np.all(np.isfinite(kl_q_latent.detach().cpu().numpy()))

loss += (class_weights[:, z] * kl_q_latent).sum()

# print('log like', loss / image.size()[0])
# kl term for latent parameters
# (assuming standard normal prior)

# print('kl q latent', kl_q_latent / image.size()[0])

def eval_semi_supervised_loss(vae, loader_unlabeled,
                        labeled_images = None, labels = None,
                        optimizer = None, train = False,
                        alpha = 1.0, num_reinforced = 0):
    if train:
        vae.train()
        assert optimizer is not None
    else:
        vae.eval()

    avg_semisuper_loss = 0.0
    avg_unlabeled_loss = 0.0

    num_unlabeled_total = loader_unlabeled.sampler.data_source.num_images

    i = 0
    for batch_idx, data in enumerate(loader_unlabeled):

        # if torch.cuda.is_available():
        unlabeled_images = data['image'].to(device)
        if labeled_images is not None:
            labeled_images = labeled_images.to(device)
            labels = labels.to(device)
        # else:
        #     unlabeled_images = data['image']

        if optimizer is not None:
            optimizer.zero_grad()

        batch_size = unlabeled_images.size()[0]

        semi_super_loss, semi_super_ps_loss, \
            unlabeled_loss, labeled_loss, \
            cross_entropy_term = \
                vae.get_semisupervised_loss(unlabeled_images,
                                            num_unlabeled_total,
                                            labeled_images = labeled_images,
                                            labels = labels,
                                            alpha = alpha,
                                            num_reinforced = num_reinforced)

        if train:
            (semi_super_ps_loss).backward()
            optimizer.step()

        avg_semisuper_loss += semi_super_loss.data / num_unlabeled_total
        avg_unlabeled_loss += unlabeled_loss.data * \
                (batch_size / num_unlabeled_total)

    return avg_semisuper_loss, avg_unlabeled_loss
