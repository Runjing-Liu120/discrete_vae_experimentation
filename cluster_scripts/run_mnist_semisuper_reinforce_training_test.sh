#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 1000 \
			--save_every 100000 \
			--batch_size 64 \
			--propn_sample 0.005 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_reinforce_warmstart' \
			--alpha 0.0 \
			--propn_labeled 0.1 \
			--reinforce True \
			--latent_dim 32 \
			--load_enc True \
			--load_dec True \
			--load_classifier True \
			--enc_init '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_enc_final' \
			--dec_init '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_dec_final' \
			--train_classifier_only True \
			--classifier_init '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_classifier_final'
			--learning_rate 0.00001	
