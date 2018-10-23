#!/bin/bash

python ../libraries/run_mnist_classifier_training.py \
			--epochs 50 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'mnist_classifier_only' \
			--propn_labeled 0.05 \
			--propn_sample 1.0 \
			--learning_rate 1e-2
