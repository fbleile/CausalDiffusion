import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Asymmetric generalized logistic function-like layers
        self.layer1 = nn.Linear(2, 64)  # 2 input features: dimension and bandwidth
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 1)
        
        # Use a non-linear activation function similar to the asymmetric logistic function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.layer1(x))  # Non-linearity after first layer
        x = self.sigmoid(self.layer2(x))  # Non-linearity after second layer
        x = self.tanh(self.layer3(x))  # Non-linearity after third layer
        x = self.layer4(x)  # Output layer
        # Apply exponential to ensure positive output, and scale for the threshold
        epsilon = 10e-8
        return torch.exp(x) + epsilon