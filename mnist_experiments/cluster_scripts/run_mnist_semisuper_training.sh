#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 300 \
			--save_every 50 \
			--batch_size 256 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/mnist_conte_vae_semisuper_1000labeled' \
			--alpha 1.0 \
			--propn_labeled 0.01666666666666666 \
			--latent_dim 32 \
			--learning_rate 1e-3 
# 			--load_enc True \
#			--load_dec True \
#			--load_classifier True \
#			--classifier_init '../mnist_vae_results_aws/mnist_conte_vae_semisuper_classifier_final' \
#                       --enc_init '../mnist_vae_results_aws/mnist_conte_vae_semisuper_enc_final' \
#                        --dec_init '../mnist_vae_results_aws/mnist_conte_vae_semisuper_classifier_final' 

