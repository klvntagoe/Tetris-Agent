import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNN(nn.Module):
    NAME = "QNN1_nn"
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu")
    
    def __init__(
            self,
            numInputs,
            numOutputs):
        super(QNN, self).__init__()
        self.layer1 = nn.Linear(numInputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, numOutputs)
    
    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out
    
    def preProcess(state):
    # Convert to a 20x10 + 4x4 = 216 length 1D tensor
        board = state['board']
        board = board[:20,4:14] # Remove padding
        board[board != 0] = 1 # Set all non-empty cells to 1
        flattenedBoard = board.astype(np.float32).flatten()
        holder = state['holder']
        holder[holder != 0] = 1
        flattenedHolder = holder.astype(np.float32).flatten()
        aggregate = np.concatenate((flattenedBoard, flattenedHolder))
        return torch.tensor(aggregate, device=QNN.device, dtype=torch.float).unsqueeze(0)