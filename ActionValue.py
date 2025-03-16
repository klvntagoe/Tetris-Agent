from collections import deque, namedtuple
from model.qnn import QNN
import os
import datetime
import numpy as np
import random as rand
import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'nextState', 'reward'))

class ReplayBuffer(object):
    def __init__(self, seed, capacity):
        # Seeding
        if seed is not None:
            rand.seed(seed)
        
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batchSize):
        return rand.sample(self.buffer, batchSize)

    def __len__(self):
        return len(self.buffer)

class ActionValueFunction:
    def __init__(
            self, 
            seed: int,
            numActions: int,
            modelPath: str = None,
            train: bool = True,
            learningRate = 1e-2,
            discountFactor = 0.99,
            replayBufferCapacity = 100_000,
            batchTransitionSampleSize = 32,
            trainingFrequency = 4,
            checkpointRate = 100_000):
        # Seeding
        self.seed = -1
        if seed is not None:
            self.seed = seed
            rand.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.targetNetwork = QNN(
            numInputs=200, 
            numOutputs=numActions).to(QNN.device)
        
        self.train = train
        if self.train:
            self.discountFactor = discountFactor
            self.optimizer = optim.AdamW(
                self.targetNetwork.parameters(), 
                lr= learningRate, 
                amsgrad=True)
            self.replayBuffer = ReplayBuffer(
                seed, 
                replayBufferCapacity)
            self.batchTransitionSampleSize = batchTransitionSampleSize
            self.trainingFrequency = trainingFrequency
            self.checkpointRate = checkpointRate
            
        if modelPath is not None:
            self._loadModel(modelPath)
        
        self.numEpisodes = 0
        self.numUpdates = 0         # Number of calls to update state info
        self.numTrainingSteps = 0   # Number of optimizations
    
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

    def update(self, state, action, reward, nextState, runTDUpdate):
        if not self.train:
            return
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
        self.numUpdates += 1

        if runTDUpdate:
            if self.numUpdates % self.trainingFrequency == 0:
                self._optimize()
                self.numTrainingSteps += 1
            if self.numEpisodes % self.checkpointRate == 0:
                self._saveModel()
    
    def signalEpisodeEnd(self):
        self.numEpisodes += 1

    def _optimize(self):
        batchSize = self.batchTransitionSampleSize
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
        expectedStateActionValues = rewardBatch + (self.discountFactor * nextStateValues)

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
        path = f'./model/{self.targetNetwork.NAME}_checkpoint_{now}.pth' if self.seed < 0 else f'./model/{self.targetNetwork.NAME}_checkpoint_{now}_seed_{self.seed}.pth'
        model = self.targetNetwork
        torch.save(
            {
                'discountFactor': self.discountFactor,      # override any discount factor input
                'numTrainingSteps': self.numTrainingSteps,
                'model_state_dict': self.targetNetwork.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),    # learning rate baked in
            }, 
            path)

    def _loadModel(self, path):
        if path is not None:
            try:
                if not os.path.exists(path):
                    print(f"Path does not exist: {path}")
                    return
                checkpoint = torch.load(path)
                self.targetNetwork.load_state_dict(checkpoint['model_state_dict'])
                if self.train:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.discountFactor = checkpoint['discountFactor']
                    self.numTrainingSteps = checkpoint['numTrainingSteps']
                print (f"Loaded model from {path}")  
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No model found to load")