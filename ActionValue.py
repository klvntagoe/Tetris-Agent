from collections import deque, namedtuple
from model.qnn import QNN
from ReplayBuffer import ReplayBuffer, Transition
import glob
import os

import datetime
import numpy as np
import random as rand
import torch
import torch.nn as nn
import torch.optim as optim

class ActionValueFunction:
    hyperParameters = None
    numActions = None
    optimizer = None
    randomVar = None
    replayBuffer = None
    targetNetwork = None

    def __init__(
            self, 
            seed : int,
            numActions : int,
            hyperParameters):
        self.randomVar = rand.Random(seed)
        self.numActions = numActions
        self.hyperParameters = hyperParameters

        self.targetNetwork = QNN(
            numInputs=200, 
            numOutputs=numActions, 
            hyperParameters=hyperParameters).to(QNN.device)
        self.optimizer = optim.AdamW(
            self.targetNetwork.parameters(), 
            lr=hyperParameters["learning_rate"], 
            amsgrad=True)
        self._loadModel()
        self.replayBuffer = ReplayBuffer(
            seed, 
            self.hyperParameters["buffer_capacity"])
        self.numEpisodes = 0
        self.numTrainingSteps = 0
        self.checkpointRate = 5000
    
    # Convert to a 20x10 = 200 length 1D tensor
    def preProcessState(self, state):
        board = state['board']
        board = board[:20,4:14] # Remove padding
        board[board != 0] = 1 # Set all non-empty cells to 1
        flattenedBoard = board.astype(np.float32).flatten()
        return torch.tensor(flattenedBoard, device=QNN.device, dtype=torch.float).unsqueeze(0)
    
    def evaluate(self, state):
        self.targetNetwork.eval()   # Set the model to evaluation mode
        with torch.no_grad():
            qValues = self.targetNetwork(self.preProcessState(state));
        return qValues.cpu().numpy();

    def update(self, state, action, reward, nextState):
        state = self.preProcessState(state)
        action = torch.tensor([[action]], device=QNN.device)
        if nextState is not None:
            nextState = self.preProcessState(nextState)
        reward = torch.tensor([reward], device=QNN.device)

        self.replayBuffer.push(
            state, 
            action, 
            nextState, 
            reward)
        self._optimize()
        self.numTrainingSteps += 1
        
        if self.numEpisodes != 0 and self.numEpisodes % self.checkpointRate == 0:    # Save the model
            self._saveModel()
    
    def signalEpisodeEnd(self):
        self.numEpisodes += 1

    def _optimize(self):
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
                device=QNN.device, 
                dtype=torch.bool)
        
        nonTerminatingNextStates = torch.cat([s for s in batch.nextState if s is not None])
        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state
        stateActionValues = self.targetNetwork(stateBatch).gather(1, actionBatch)

        # Compute V(s_{t+1}) for all non-terminal next states
        nextStateValues = torch.zeros(batchSize, device=QNN.device)
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

    def _saveModel(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        path = f'./model/{self.targetNetwork.NAME}_checkpoint_{now}.pth'
        model = self.targetNetwork
        torch.save(
            {
                'numEpisodes': self.numEpisodes,
                'numTrainingSteps': self.numTrainingSteps,
                'model_state_dict': self.targetNetwork.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, 
            path)

    
    def _loadModel(self):
        model_files = glob.glob(f'/model/{self.targetNetwork.NAME}*.pth')
        latest_model = max(model_files, key=os.path.getctime) if model_files else None
        if latest_model:
            self.numEpisodes = checkpoint['numEpisodes']
            self.numTrainingSteps = checkpoint['numTrainingSteps']
            checkpoint = torch.load(latest_model)
            self.targetNetwork.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print (f"Loaded state from {latest_model}")
        else:
            print("No model state found to load")