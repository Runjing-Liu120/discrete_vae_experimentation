#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 201 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--topk 0 \
			--n_samples 6 \
			--vae_outdir '../galaxy_results_sandbox/reinforce_nsamples6/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_reinforce_nsamples6' 

