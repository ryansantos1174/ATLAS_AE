import torch.nn as nn
from fastai import learner
import time
import os
import shutil
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self, n_features,latent, neurons, act):
        super(AE, self).__init__()
        self.en1 = nn.Linear(n_features, neurons[0])
        self.en2 = nn.Linear(neurons[0],neurons[1])
        self.en3 = nn.Linear(neurons[1],neurons[2])
        self.en4 = nn.Linear(neurons[2],latent)
        self.de1 = nn.Linear(latent, neurons[2])
        self.de2 = nn.Linear(neurons[2], neurons[1])
        self.de3 = nn.Linear(neurons[1],neurons[0])
        self.de4 = nn.Linear(neurons[0], n_features)
        self.activation = act
        
    def encode(self, x):
        return self.en4(self.activation(self.en3(self.activation(self.en2(self.activation(self.en1(x)))))))
       
    def decode(self, x):
        return self.de4(self.activation(self.de3(self.activation(self.de2(self.activation(self.de1(self.activation(x))))))))
           
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


    