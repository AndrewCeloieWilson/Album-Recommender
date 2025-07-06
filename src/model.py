import torch.nn as nn

'''
    Defines a feedforward neural network for binary classification of song preferences.

    The architecture consists of several fully connected (dense) layers with ReLU activations,
    dropout regularization, and a final sigmoid activation to output a probability between 0 and 1.

    This model is designed to handle structured audio feature input data, such as those
    extracted from AcousticBrainz, and predict whether a user will like a given song.

    Args:
        input_dim (int): The number of input features (columns in the training data).

    Methods:
        forward(x): Defines the forward pass of the network using the defined layers.
'''
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            # First layer: neurons + ReLU + Dropout
            nn.Linear(input_dim, 132),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.0),

            # Second hidden layer
            nn.Linear(132, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),

            # Third hidden layer
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),

            # Fourth hidden layer
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),
            
            # Output layer: 1 neuron + Sigmoid (for binary classification)
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    '''
        Forward pass: applies the sequence of layers to input x.
    '''
    def forward(self, x):
        return self.net(x)