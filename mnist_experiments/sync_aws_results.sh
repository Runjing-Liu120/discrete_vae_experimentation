rsync -avL --progress -e 'ssh -i ./../../bryans_key_oregon.pem' \
   ubuntu@ec2-54-187-164-254.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/discrete_vae_experimentation/mnist_vae_results_aws/.\
   /home/runjing_liu/Documents/astronomy/discrete_vae_experimentation/mnist_vae_results_aws/.

