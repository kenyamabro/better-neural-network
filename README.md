This project implements a Fully Connected Neural Network with the algorithm built from scratch, without using high-level ML libraries. 
It uses a tkinter GUI to customize these 3 parameters : hidden layers (number of layers and number of neurons per layer), learning rate, and batch size. 
The network is trained on the MNIST dataset.
After training, the following is displayed (the plots are pyplots from matplotlib):
  - Plot of accuracy and average cost per output neuron vs. batch number
  - Heatmaps of the weights between each second-layer neuron and all input neurons
