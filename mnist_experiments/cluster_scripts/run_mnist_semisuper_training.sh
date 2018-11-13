#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 200 \
			--save_every 10 \
			--batch_size 256 \
			--seed 901 \
			--outdir '../mnist_vae_results/' \
			--outfilename 'mnist_vae_warm_start_classifier' \
			--alpha 1.0 \
			--topk 10 \
			--use_baseline False \
			--propn_labeled 0.05 \
			--propn_sample 1.0 \
			--latent_dim 8 \
			--learning_rate 1e-3 \
			--use_true_labels False \
                        --load_classifier True \
                        --classifier_init '../mnist_vae_results/mnist_vae_true_labels_classifier_final'

