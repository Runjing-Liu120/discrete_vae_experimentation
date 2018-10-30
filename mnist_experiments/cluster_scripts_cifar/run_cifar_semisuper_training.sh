#!/bin/bash

python ../libraries/run_cifar_semisupervised_training.py \
			--epochs 50 \
			--save_every 10 \
			--batch_size 256 \
			--seed 901 \
			--outdir '../cifar_vae_results/' \
			--outfilename 'mnist_vae_true_labels' \
			--alpha 1.0 \
			--topk 0 \
			--use_baseline True \
			--propn_labeled 0.005 \
			--propn_sample 1.0 \
			--learning_rate 1e-2 \
			--use_true_labels True \
                        --load_classifier False \
                        --classifier_init '../mnist_vae_results/mnist_vae_true_labels_classifier_final'

