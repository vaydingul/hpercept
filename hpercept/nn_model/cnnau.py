import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
# Define the Convolutional Autoencoder


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 4, (4, 10), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 16, (2, 2), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 20, (2, 2), stride=2, padding=1),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(360, 3),
            nn.ReLU(),
            nn.Linear(3, 360),
            nn.Unflatten(1, (20, 3, 6))
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(20, 16, (2, 2), stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, (2, 2), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(12, 8, (3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, (3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, (2, 2), stride=2,
                               padding=4, output_padding=0),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
    

"""
#Loss function
criterion = nn.BCELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
"""
