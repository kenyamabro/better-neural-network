This project implements a Fully Connected Neural Network with the algorithm built from scratch (using some basic numpy functions), without using high-level ML libraries.
The network is trained on the MNIST dataset (60000 28x28 pixels images of handwritten digits).
The home window has entries to customize these 5 parameters : hidden layers (number of layers and number of neurons per layer), batch number, batch size, learning rate and noise.
  
After training, the neural network is stored on a listbox.
Clicking on a neural network in the list will display the following:
  - Drawing interface where the neural network guesses the digit that is drawn (the drawing interface is built from scratch too)
  - Plot of accuracy and average cost per output neuron vs. batch number
  - Heatmaps of the weights between each second-layer neuron and all input neurons (visual representation of the weights attributed to each pixel in the images by each second-layer neuron)

Interface libraries used:
  - Tkinter in general
  - Matplotlib for the plots

Demo:
![This is a demo of the application](https://github.com/kenyamabro/better-neural-network/blob/main/demo.png)
