rsync -avL --progress -e 'ssh -i ./../../../bryans_key_oregon.pem' \
   ubuntu@ec2-52-38-247-127.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/discrete_vae_experimentation/mnist_experiments/mnist_vae_results/.\
   /home/runjing_liu/Documents/astronomy/discrete_vae_experimentation/mnist_experiments/mnist_vae_results/.

