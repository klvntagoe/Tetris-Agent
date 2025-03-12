from collections import deque, namedtuple
import datetime
from ReplayBuffer import ReplayBuffer, Transition
import numpy as np
import random as rand
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNN(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # if GPU is to be used
    NAME = "QNN1_nn"

    def __init__(
            self,
            numInputs,
            numOutputs,
            hyperParameters):
        super(QNN, self).__init__()
        self.hyperParameters = hyperParameters
        self.layer1 = nn.Linear(numInputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, numOutputs)
    
    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out