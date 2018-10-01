rsync -avL --progress -e 'ssh -i ./../../../bryans_key_oregon.pem' \
   ubuntu@ec2-34-223-4-100.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/discrete_vae_experimentation/partial_marginalization_experiments/galaxy_results/.\
   /home/runjing_liu/Documents/astronomy/discrete_vae_experimentation/partial_marginalization_experiments/galaxy_results/.

