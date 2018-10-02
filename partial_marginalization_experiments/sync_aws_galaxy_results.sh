rsync -avL --progress -e 'ssh -i ./../../../bryans_key_oregon.pem' \
   ubuntu@ec2-54-71-18-68.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/discrete_vae_experimentation/partial_marginalization_experiments/galaxy_results/.\
   /home/runjing_liu/Documents/astronomy/discrete_vae_experimentation/partial_marginalization_experiments/galaxy_results/.

