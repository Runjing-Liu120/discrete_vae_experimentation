#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--slen 20 \
			--epochs 2001 \
			--save_every 100 \
			--batchsize 64 \
			--seed 904 \
			--topk 10 \
			--n_samples 1 \
			--vae_outdir '../galaxy_results_sandbox2/topk10/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_topk10' 
			
