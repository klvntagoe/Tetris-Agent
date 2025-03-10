from collections import deque, namedtuple
from ReplayBuffer import ReplayBuffer, Transition
import numpy as np
import random as rand
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # if GPU is to be used

    def __init__(
            self,
            numInputs,
            numOutputs,
            hyperParameters):
        super(QNetwork, self).__init__()
        self.hyperParameters = hyperParameters
        self.layer1 = nn.Linear(numInputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, numOutputs)
    
    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out

class TetrisActionValueNetwork:
    randomVar = None
    numActions = None
    targetNetwork = None
    optimizer = None
    replayBuffer = None
    hyperParameters = None

    def __init__(
            self, 
            seed : int,
            numActions : int,
            hyperParameters):
        self.randomVar = rand.Random(seed)
        self.numActions = numActions
        self.hyperParameters = hyperParameters
        self.targetNetwork = QNetwork(
            numInputs=200, 
            numOutputs=numActions, 
            hyperParameters=hyperParameters).to(QNetwork.device)
        self.optimizer = optim.AdamW(self.targetNetwork.parameters(), 
                                     lr=hyperParameters["learning_rate"], 
                                     amsgrad=True)
        self.replayBuffer = ReplayBuffer(seed, 100)

    # Convert to a 20x10 = 200 length 1D tensor
    def preProcessState(self, state):
        board = state['board']
        board = board[:20,4:14] # Remove padding
        board[board != 0] = 1 # Set all non-empty cells to 1
        flattenedBoard = board.astype(np.float32).flatten()
        return torch.tensor(flattenedBoard, device=QNetwork.device, dtype=torch.float).unsqueeze(0)
    
    def evaluate(self, state):
        self.targetNetwork.eval()   # Set the model to evaluation mode
        with torch.no_grad():
            qValues = self.targetNetwork(self.preProcessState(state));
        return qValues.cpu().numpy();

    def updateActionValue(self, state, action, reward, nextState):
        state = self.preProcessState(state)
        action = torch.tensor([[action]], device=QNetwork.device)
        if nextState is not None:
            nextState = self.preProcessState(nextState)
        reward = torch.tensor([reward], device=QNetwork.device)
        self.replayBuffer.push(state, action, nextState, reward)
        self.optimize()

    def optimize(self):
        batchSize = self.hyperParameters["batch_size"]
        if len(self.replayBuffer) < batchSize:
            return
        
        self.targetNetwork.train()  # Set the model to training mode

        transitions = self.replayBuffer.sample(batchSize)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). 
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        nonFinalMask = torch.tensor(
                tuple(map(lambda s: s is not None,batch.nextState)),
                device=QNetwork.device, 
                dtype=torch.bool)
        
        nonTerminatingNextStates = torch.cat([s for s in batch.nextState if s is not None])
        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state
        stateActionValues = self.targetNetwork(stateBatch).gather(1, actionBatch)

        # Compute V(s_{t+1}) for all non-terminal next states
        nextStateValues = torch.zeros(batchSize, device=QNetwork.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = self.targetNetwork(nonTerminatingNextStates).max(1).values
        
        # Compute the expected Q values
        gamma = self.hyperParameters["gamma"]
        expectedStateActionValues = rewardBatch + (gamma * nextStateValues)

        # Compute TD error using Huber Loss 
        lossCriterion = nn.SmoothL1Loss()
        tdHuberError = lossCriterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        tdHuberError.backward()
        # torch.nn.utils.clip_grad_norm_(self.targetNetwork.parameters(), 100)    # Clip the gradient in-place
        self.optimizer.step()