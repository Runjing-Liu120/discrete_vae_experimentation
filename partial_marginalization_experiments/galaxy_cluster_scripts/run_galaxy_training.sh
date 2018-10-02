#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 501 \
			--save_every 5 \
			--batchsize 64 \
			--seed 9045345 \
			--topk 0 \
			--n_samples 1 \
			--vae_outdir '../galaxy_results/reinforce_nsamples1a/' \
                        --vae_warm_start True \
                        --vae_init_file '../galaxy_results/reinforce_nsamples1/galaxy_vae_reinforce_nsamples1_epoch80.dat'\
			--vae_outfilename 'galaxy_vae_reinforce_nsamples1' 

