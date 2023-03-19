# Import libraries
import sys
import numpy as np
from math import sqrt


# Create the neural network class
class MultiLayerPerceptron():
    '''
    Note that this neural network will be trained to predict continuous variable:
    - Sigmoid will be used as activation function, and 
    - Normalized Xavier initialization will be used to initialize weights of the layers
    '''
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, iterations, batch_size, activation_function, val_ratio=0.8):
        # Set the number of nodes in each of the 3 layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Set the learning rate
        self.learning_rate = learning_rate

        # Set the iterationsation count
        self.iterations = iterations

        # Set the batch size
        self.batch_size = batch_size 

        # Set the activation function
        self.activation_function_name = activation_function
        self.activation_function = self.activation_function_name

        # Set the validation dataset proportion
        self.val_ratio = val_ratio

        # Initialize the weights using Normalized Xavier Weight Initialization method
        input_to_hidden_floor = -(sqrt(6.0) / sqrt(self.input_nodes + self.hidden_nodes))
        input_to_hidden_ceiling = -1 * input_to_hidden_floor
        hidden_to_output_floor = -(sqrt(6.0) / sqrt(self.hidden_nodes + self.output_nodes))
        hidden_to_output_ceiling = -1 * hidden_to_output_floor

        # Initialize weights from input layer to hidden layer
        self.weights_input_to_hidden = np.random.uniform(
            input_to_hidden_floor,
            input_to_hidden_ceiling,
            (self.input_nodes, self.hidden_nodes)
        )
        
        # Initialize weights from hidden layer to output layer
        self.weights_hidden_to_output = np.random.uniform(
            hidden_to_output_floor,
            hidden_to_output_ceiling,
            (self.hidden_nodes, self.output_nodes)
        )

        # Sigmoid activation function
        def sigmoid(x):
            '''
            Args:
            x (array) - An array of continuous values
            
            Returns:
            An array that have been transformed using sigmoid formula

            '''
            return np.exp(x) / (1 + np.exp(x))


        # Tanh activation function
        def tanh(x):
            '''
            Args:
            x (array) - An array of continuous values
            
            Returns:
            An array that have been transformed using tanh formula

            '''
            return (np.exp(2 * x) - 1)/ (np.exp(2 * x) + 1)
        

        # Relu activation function
        def relu(x):
            '''
            Args:
            x (array) - An array of continuous values
            
            Returns:
            An array that have been transformed using relu formula

            '''
            return np.maximum(0, x)
        

        # Select activation function
        if self.activation_function_name == 'sigmoid':
            self.activation_function = sigmoid
        
        elif self.activation_function_name == 'tanh':
            self.activation_function = tanh

        elif self.activation_function_name == 'relu':
            self.activation_function = relu


    # Feed forward pass function
    def feed_forward_pass(self, X):
        '''
        Args:
        X (matrix) - Matrix of input values of shape (n, input_nodes)
        
        Returns:
        hidden_outputs (matrix) - Matrix of hidden layer outputs of shape (n, hidden_nodes)
        final_outputs (matrix) - Matrix of output layer outputs of shape (n, output_nodes)
        
        ''' 
        # Pass from input layer to hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Pass from hidden layer to output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return hidden_outputs, final_outputs
    

    # Backpropagation function
    def back_propagation(self, hidden_outputs, final_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
        Args:
        hidden_outputs (matrix) - Matrix of hidden layer outputs of shape (n, hidden_nodes)
        final_outputs (matrix) - Matrix of output layer outputs of shape (n, output_nodes)
        X (matrix) - Matrix of input values of shape (n, input_nodes)
        y (array) - An array of target variable values of shape (n, output_nodes)
        
        Returns:
        delta_weights_i_h (matrix) - Matrix of deltas for input to hidden matrix of shape (input_nodes, hidden_nodes)
        delta_weights_h_o (matrix) - Matrix of deltas for hidden to output matrix of shape (hidden_nodes, output_nodes)
        
        ''' 
        # Gradient Descent Method
        # Calculate the error at the output later
        error = y - final_outputs

        # Output error term
        output_error_term = error

        # Calculate the error contributed by hidden layer
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)

        # Hidden error term
        # For sigmoid function, f'(h) = f(h) * (1 - f(h))
        # For tanh function, f'(h) = 1 - f(h)**2
        # For relu function, f'(h) = 1 if x >= 0 else 0
        if self.activation_function_name == 'sigmoid':
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        elif self.activation_function_name == 'tanh':
            hidden_error_term = hidden_error * (1 - hidden_outputs ** 2)

        elif self.activation_function_name == 'relu':
            hidden_error_term = self.activation_function(hidden_error)
    
        # Weight step (hidden to output)
        delta_weights_h_o += hidden_outputs[:, None] * output_error_term
        
        # Weight step (input to hidden)
        delta_weights_i_h += X[:, None] * hidden_error_term
        
        return delta_weights_i_h, delta_weights_h_o
    

    # Update Weights
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        '''
        Args:
        delta_weights_i_h (matrix) - Matrix of deltas for input to hidden matrix of shape (input_nodes, hidden_nodes)
        delta_weights_h_o (matrix) - Matrix of deltas for hidden to output matrix of shape (hidden_nodes, output_nodes)
        n_records (int) - Number of records

        Returns:
        self.weights_input_to_hidden (updated)
        self.weights_hidden_to_output (updated)
        '''
        # Update the input to hidden weights
        self.weights_input_to_hidden += delta_weights_i_h * self.learning_rate / n_records

        # Update the hidden to output weights
        self.weights_hidden_to_output += delta_weights_h_o * self.learning_rate / n_records


    # Model training
    def train(self, features, targets):
        '''
        Args:
        features (matrix): Input matrix of shape (n_records, input_nodes)
        target (matrix): Target matrix of shape (n_records, output_nodes) 
        '''
        # Determine number of records
        n_records = features.shape[0]

        # Initialize delta weights
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        # Perform model training
        for X, y in zip(features, targets):
        
            # Forward pass
            hidden_outputs, final_outputs = self.feed_forward_pass(X)
            
            # Back-propagation
            delta_weights_i_h, delta_weights_h_o = self.back_propagation(
                hidden_outputs, 
                final_outputs, 
                X, 
                y, 
                delta_weights_i_h, 
                delta_weights_h_o

            )
        
        # Update weights
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
    

    # Calculate the mean squared error between predicted and actual values
    def MSE(self, y, Y):
        '''
        Args:
        y (matrix) - Actual observed target values of shape (input_nodes, hidden_nodes)
        Y (matrix) - Predicted target values of shape (input_nodes, hidden_nodes)

        Returns:
        mean squared error (float)
        '''
        return np.mean((y - Y) ** 2)
    

    # Model fitting
    def fit(self, features, targets):
        '''
        Args:
        features (matrix): Input matrix of shape (n_records, input_nodes)
        target (matrix): Target matrix of shape (n_records, output_nodes) 

        Returns:
        train_losses (list): List of training losses of length iterations
        '''            
        # Initialize losses
        train_losses = []

        # Loop through number of iterations
        for iter in range(self.iterations):
            # Go through a random batch from the data 
            batch = np.random.choice(features.index, size=self.batch_size)
            X, y = features.iloc[batch].values, targets.iloc[batch].values
            
            # Perform model training
            self.train(X, y)
            
            # Predict & calculate MSE
            _, y_pred = self.feed_forward_pass(X)
            train_loss = self.MSE(y_pred, y)
            
            # Print out the training progress
            sys.stdout.write('\rProgress: {:2.1f}'.format(100 * iter / float(self.iterations)) \
                            + '% ... Training loss: ' + str(train_loss)[:5])
            sys.stdout.flush()
            
            # Append loss to dictionary
            train_losses += [train_loss]

        return train_losses