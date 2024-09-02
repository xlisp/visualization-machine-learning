# 	symbol_derivative_pytorch.png : Implement this algorithm using pytorch

import torch
import torch.nn.functional as F


class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, layers):
        super(SimpleNeuralNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            a = layer(h)
            h = F.relu(a)  # Assuming f(a) is ReLU activation
        # No activation for the last layer
        y_hat = self.layers[-1](h)
        return y_hat

# Define the loss function
def loss_function(y_hat, y, model, lambda_reg=0.01):
    # Assuming a mean squared error loss for simplicity
    loss = F.mse_loss(y_hat, y)
    # Adding L2 regularization
    l2_reg = lambda_reg * sum(torch.norm(param) for param in model.parameters())
    return loss + l2_reg

# Example usage
input_dim = 10
output_dim = 1
hidden_layers = [5, 5]  # Define the hidden layer sizes

# Define the model with input, hidden, and output layers
model = SimpleNeuralNetwork([input_dim] + hidden_layers + [output_dim])

# Example data
x = torch.randn(10)  # Input tensor
y = torch.randn(1)   # Target tensor

# Forward pass
y_hat = model(x)

# Compute loss
J = loss_function(y_hat, y, model)

# Print the loss
print("Loss J:", J.item())

# python symbol_derivative_pytorch.py
# Loss J: 0.04097992181777954
# emacspy-machine-learning  master @ python symbol_derivative_pytorch.py
# Loss J: 0.1827695369720459
# emacspy-machine-learning  master @ python symbol_derivative_pytorch.py
# Loss J: 0.14362196624279022

