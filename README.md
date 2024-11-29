This project implements a Fully Connected Neural Network with the algorithm built from scratch, without using high-level ML libraries. 
The network is trained on the MNIST dataset.
It uses a tkinter GUI to customize these 4 parameters : hidden layers (number of layers and number of neurons per layer), learning rate, batch size and noise. 
After training, the neural network is stored on a list.
Clicking on a neural network in the list will display the following (the plots are pyplots from matplotlib):
  - Drawing interface where the neural network guesses the digit that is drawn.
  - Plot of accuracy and average cost per output neuron vs. batch number
  - Heatmaps of the weights between each second-layer neuron and all input neurons

Demo:
![This is a demo of the application](https://github.com/kenyamabro/better-neural-network/blob/main/demo.png)
