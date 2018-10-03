def sample_class_weights(class_weights, num_reinforced):
    # TODO: THIS FUNCTION NEEDS TO BE TESTED

    len_sample_domain = len(class_weights)
    assert num_reinforced <= len_sample_domain

    # fudge factor to break ties in probabilities
    # fudge_factor = 0.0 # 1e-6 * torch.rand(len_sample_domain)

    # get the k smallest weights, and the indices to which they correspond
    # we will be sampling from these weights
    sampled_weight, sampled_z_domain = torch.topk(-class_weights, num_reinforced)
    sampled_weight = -sampled_weight

    # in the future, we also need the indices not sampled
    if not num_reinforced == len(class_weights):
        unsampled_z_domain = torch.LongTensor([i for i in range(len(class_weights)) if
                                      i not in sampled_z_domain])
        unsampled_weight = class_weights[unsampled_z_domain]

        # seq = torch.LongTensor([[i for i in range(len(class_weights))])
        #
        # unsampled_z_domain = torch.LongTensor([seq[i] for i in range(len(seq))])

        # unsampled_weight, unsampled_z_domain = torch.topk(class_weights + fudge_factor,
        #                                                   len_sample_domain - num_reinforced)
        # unsampled_weight = unsampled_weight - fudge_factor[unsampled_z_domain]
    else:
        unsampled_z_domain = []
        unsampled_weight = torch.Tensor([0.0])

    assert len(np.intersect1d(sampled_z_domain.numpy(), unsampled_z_domain.numpy())) == 0

    # sample
    # print(sampled_weight / torch.sum(sampled_weight))
    sample_probs = (sampled_weight + 1e-6)/ torch.sum(sampled_weight + 1e-6)

    cat_rv = Categorical(probs = sample_probs)
    z_sample = sampled_z_domain[cat_rv.sample().detach()]

    return z_sample.detach(), sampled_z_domain, sampled_weight, \
                unsampled_z_domain, unsampled_weight
