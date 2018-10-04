#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 501 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--topk 0 \
			--n_samples 4 \
			--vae_outdir '../galaxy_results/testing_star_proto_reinforce4/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_reinforce4' 

