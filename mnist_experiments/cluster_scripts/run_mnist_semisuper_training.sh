#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 50 \
			--save_every 10 \
			--batch_size 256 \
			--seed 901 \
			--outdir '../mnist_vae_results/' \
			--outfilename 'mnist_vae_topk10' \
			--alpha 1.0 \
			--topk 10 \
			--use_baseline False \
			--propn_labeled 0.05 \
			--propn_sample 1.0 \
			--latent_dim 4 \
			--learning_rate 1e-2 \
			--use_true_labels False \
                        --load_classifier True \
                        --classifier_init '../mnist_vae_results/mnist_vae_true_labels_classifier_final'

