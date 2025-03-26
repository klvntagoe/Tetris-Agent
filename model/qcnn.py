import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Q_CNN(nn.Module):
    NAME = "QNetwork_cnn1"
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu")
    
    def __init__(self,numOutputs):
        super().__init__()
        self.network = nn.Sequential(
        # Feature extraction block 1
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,5), stride=1, padding=1, groups=4), # depth-wise separated convolution over each channel
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.GroupNorm(num_groups=4, num_channels=32),  # noamlize channels independently
            nn.ReLU(),
        # Feature extraction block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=1, padding=0, groups=4), # depth-wise separated convolution over the filter set of each original channel
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.GroupNorm(num_groups=32, num_channels=64),  # one group per original channel
            nn.ReLU(),
        # Dense block
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, numOutputs)
        )
    
    def forward(self, preProcessedInput):
        return self.network(preProcessedInput)
    
    def preProcess(input):
        # Remove environment padding of board and active tetrimino mask
        # Converts dimensions from 24x18 to 20x10 by removing 4 rows on the left and right and bottom 4 columns
        board = input['board']
        board = board[:20,4:14]
        mask = input['active_tetromino_mask']
        mask = mask[:20,4:14]

        # Convert board to binary signal of empty/filled pixels
        board[board != 0] = 1 # Set all non-empty cells to 1

        # Rotate queue counter clock-wise
        # Pad queue on both sides by 3 and above and below by 2 to convert dimensions from 4x16 to 20x10
        queue = input['queue']
        queue = np.rot90(queue)

        zeroes = np.zeros((queue.shape[0], 1), dtype=queue.dtype)   # 16x1
        queue = np.concatenate([
            zeroes, zeroes, zeroes, 
            queue, 
            zeroes, zeroes, zeroes], 
            axis=1) # 16x10
        
        zeroes = np.zeros((1, queue.shape[1]), dtype=queue.dtype)   #1x10
        queue = np.concatenate([
            zeroes, zeroes, 
            queue, 
            zeroes, zeroes], 
            axis=0) #20x10
        # Set all pixels to 0 when holder array is empty (environment sets all pixels to 1 when empty)
        # Pad holder by 3 on each size and 8 above and below to convert dimensions from 4x4 to 20x10
        holder = input['holder']

        allOnes = True
        for row in holder:
            for el in row:
                if el != 1:
                    allOnes = False
                    break
        
        if allOnes:
            holder.fill(0)
        
        zeroes = np.zeros((1, holder.shape[0]), dtype=holder.dtype)   # 1x4
        holder = np.concatenate([
            zeroes, zeroes, zeroes, zeroes, zeroes, zeroes, zeroes, zeroes, 
            holder, 
            zeroes, zeroes, zeroes, zeroes, zeroes, zeroes, zeroes, zeroes], 
            axis=0) # 20x4
        
        zeroes = np.zeros((holder.shape[0], 1), dtype=holder.dtype)   # 20x1
        holder = np.concatenate([
            zeroes, zeroes, zeroes, 
            holder, 
            zeroes, zeroes, zeroes], 
            axis=1) # 20x10
        
        # Stack channels and unsqueeze so shape becomes (1, 4, 20, 10)
        return torch.tensor(np.array([board, mask, queue, holder]), device=Q_CNN.device, dtype=torch.float).unsqueeze(0)