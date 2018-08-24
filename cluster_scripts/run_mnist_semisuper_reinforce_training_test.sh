#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 1000 \
			--save_every 100000 \
			--batch_size 64 \
			--propn_sample 0.005 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/foo' \
			--alpha 0.0 \
			--propn_labeled 0.1 \
			--reinforce True \
			--use_baseline True \
			--latent_dim 32 \
			--load_enc True \
			--load_dec True \
			--load_classifier False \
			--enc_init '../saved_vae_results/mnist_vae_semisuper_enc_final' \
			--dec_init '../saved_vae_results/mnist_vae_semisuper_dec_final' \
			--classifier_init '../saved_vae_results/mnist_vae_semisuper_classifier_final'\
                        --train_classifier_only True 

