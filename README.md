## Overview

The main objective of this code is to understand the underlying components of neural network. This is quite a limited neural network that hopes to improve over time with more features. Below are the available features so far:
  - Optimizer: Gradient Descent
  - Activation Functions:
    - Sigmoid
    - Tanh
    - Relu
  - Initialization: Normalized Xavier Weight Initialization
  - Loss function: MSE
  - Hidden Layer: Only 1

## Main Functions

The class object MultiLayerPerceptron has the following key functions:
  - feed_forward_pass: Generate predicted outcomes using features dataset
  - back_propagation: Perform back propagation to optimize the neural networks' weights
  - update_weights: Update the neural networks' weights based on back propagation
  - train: Perform model training using the 3 functions listed above
  - fit: Perform model fitting by training over selected number of iterations

## Installation

Clone this git repository to your local workspace.
`git clone https://github.com/fucheng96/neural-net.git`
