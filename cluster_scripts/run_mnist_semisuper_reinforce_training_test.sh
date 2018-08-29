#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 200 \
			--save_every 50 \
			--batch_size 256 \
			--propn_sample 0.01 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/experimenting_mix_classic_reinforce/foo' \
			--alpha 0.0 \
			--propn_labeled 0.1 \
			--num_reinforced 10 \
			--use_baseline True \
			--latent_dim 32 \
			--load_enc True \
			--load_dec True \
			--load_classifier False \
			--enc_init '../saved_vae_results/mnist_vae_semisuper_enc_final' \
			--dec_init '../saved_vae_results/mnist_vae_semisuper_dec_final' \
			--classifier_init '../mnist_vae_results_aws/reinforce_classifier_warmstart_classifier_final'\
                        --train_classifier_only True \
			--learning_rate 1e-3

