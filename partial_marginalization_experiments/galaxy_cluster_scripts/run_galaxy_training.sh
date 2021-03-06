#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 31 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--topk 1 \
			--n_samples 1 \
			--vae_outdir '../galaxy_results_cv_experiments/topk1/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_topk1'\
