# Simple Generative Adversarial Network (GAN) #

This repository contains the implementation of a GAN  in numpy and explicitly not using
frameworks like TensorFlow and Pytorch. 

The task the network is designed for is also quite simple. The data base consists 
of a randomly generated point cloud (sampled from normal distributions) where each point 
belongs to one class. Using the network new points should be generated fitting the 
distributions of the original ones as best as possible.

Some results can be found in the notebook notebooks/gan_training. 
