#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 200 \
			--save_every 10 \
			--batch_size 256 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_reinforce_warmstart' \
			--alpha 1.0 \
			--propn_labeled 0.1 \
			--reinforce True \
			--latent_dim 32 \
			--load_enc True \
			--load_dec True \
			--load_classifier True \
			--enc_init '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_enc_final' \
			--dec_init '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_dec_final' \
			--classifier_init '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_classifier_final' 

