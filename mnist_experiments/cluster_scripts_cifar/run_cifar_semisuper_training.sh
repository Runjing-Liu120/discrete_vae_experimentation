#!/bin/bash

python ../libraries/run_cifar_semisupervised_training.py \
			--use_cifar100 False \
			--epochs 50 \
			--save_every 10 \
			--batch_size 32 \
			--seed 901 \
			--outdir '../cifar_vae_results/' \
			--outfilename 'testing_vae_cifar10_alllabeled' \
			--alpha 1.0 \
			--topk 0 \
			--use_baseline True \
			--propn_labeled 1.0 \
			--propn_sample 1.0 \
			--learning_rate 3e-4 \
			--use_true_labels True \
                        --load_classifier False \
                        --classifier_init '../mnist_vae_results/mnist_vae_true_labels_classifier_final'

