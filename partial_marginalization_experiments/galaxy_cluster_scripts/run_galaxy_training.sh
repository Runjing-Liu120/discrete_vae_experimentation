#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 501 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--topk 3 \
			--n_samples 1 \
			--vae_outdir '../galaxy_results/testing_star_proto_topk3/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_topk3'\
		
