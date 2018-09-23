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
