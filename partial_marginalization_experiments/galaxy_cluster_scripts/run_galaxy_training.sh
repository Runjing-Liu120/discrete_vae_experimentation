#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--slen 20 \
			--epochs 2001 \
			--save_every 100 \
			--batchsize 64 \
			--seed 904 \
			--topk 3 \
			--n_samples 1 \
			--vae_outdir '../galaxy_results_sandbox/topk3/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_topk3' 
			
