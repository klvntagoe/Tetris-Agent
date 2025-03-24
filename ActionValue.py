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
            targetNetworkUpdateFrequency = 10_000,
            checkpointRate = 1_000_000):
        # Seeding
        self.seed = -1
        if seed is not None:
            self.seed = seed
            rand.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        numInputs = 216 # Todo: make this configurable
        self.onlineNetwork = QNN(
            numInputs=numInputs, 
            numOutputs=numActions).to(QNN.device)
        
        self.train = train
        if self.train:
            self.optimizer = optim.AdamW(
                self.onlineNetwork.parameters(), 
                lr= learningRate, 
                amsgrad=True)
            self.targetNetwork = QNN(
                numInputs=numInputs, 
                numOutputs=numActions).to(QNN.device)
            self._hardUpdateTarget()
            self.targetNetworkUpdateFrequency = targetNetworkUpdateFrequency
            self.discountFactor = discountFactor
            self.replayBuffer = ReplayBuffer(
                seed, 
                replayBufferCapacity)
            self.batchTransitionSampleSize = batchTransitionSampleSize
            self.trainingFrequency = trainingFrequency
            self.checkpointRate = checkpointRate
            
        if modelPath is not None:
            self._loadModel(modelPath)
        
        self.numUpdates = 0         # Number of calls to update state info
        self.numTrainingSteps = 0   # Number of optimizations
    
    # Convert to a 20x10 + 4x4 = 216 length 1D tensor
    def preProcessState(self, state):
        board = state['board']
        board = board[:20,4:14] # Remove padding
        board[board != 0] = 1 # Set all non-empty cells to 1
        flattenedBoard = board.astype(np.float32).flatten()
        holder = state['holder']
        holder[holder != 0] = 1
        flattenedHolder = holder.astype(np.float32).flatten()
        aggregate = np.concatenate((flattenedBoard, flattenedHolder))
        return torch.tensor(aggregate, device=QNN.device, dtype=torch.float).unsqueeze(0)
    
    def evaluate(self, state):
        self.onlineNetwork.eval()   # Set the model to evaluation mode
        with torch.no_grad():
            qValues = self.onlineNetwork(self.preProcessState(state));
        return qValues.cpu().numpy();

    def update(self, state, action, reward, nextState, runTDUpdate):
        if not self.train:
            return
        state = self.preProcessState(state)
        action = torch.tensor([[action]], device=QNN.device)
        nextState = self.preProcessState(nextState) if nextState is not None else None
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
            if self.numTrainingSteps % self.targetNetworkUpdateFrequency == 0:      # save after periodic intervals of optimization steps
                self._hardUpdateTarget()
            if self.numTrainingSteps % self.checkpointRate == 0:      # save after periodic intervals of optimization steps
                self._saveModel()

    def _optimize(self):
        batchSize = self.batchTransitionSampleSize
        if len(self.replayBuffer) < batchSize:
            return
        self.onlineNetwork.train()  # Set the model to training mode
        transitions = self.replayBuffer.sample(batchSize)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). 
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        nonFinalMask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.nextState)),
                device=QNN.device, 
                dtype=torch.bool)
        
        nonTerminatingNextStates = torch.cat([s for s in batch.nextState if s is not None])
        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state
        stateActionValues = self.onlineNetwork(stateBatch).gather(1, actionBatch)

        # Compute V(s_{t+1}) for all non-terminal next states USING DELAYED TARGET NETWORK
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
        # torch.nn.utils.clip_grad_norm_(self.onlineNetwork.parameters(), 100)    # Clip the gradient in-place
        self.optimizer.step()
    
    def _hardUpdateTarget(self):
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
    
    def _saveModel(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        path = f'./model/{self.onlineNetwork.NAME}_checkpoint_{now}.pth' if self.seed < 0 else f'./model/{self.onlineNetwork.NAME}_checkpoint_{now}_seed_{self.seed}.pth'
        model = self.onlineNetwork
        torch.save(
            {
                'discountFactor': self.discountFactor,      # override any discount factor input
                'numTrainingSteps': self.numTrainingSteps,
                'model_state_dict': self.onlineNetwork.state_dict(),
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
                self.onlineNetwork.load_state_dict(checkpoint['model_state_dict'])
                if self.train:
                    self.targetNetwork.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.discountFactor = checkpoint['discountFactor']
                    self.numTrainingSteps = checkpoint['numTrainingSteps']
                print (f"Loaded model from {path}")  
                numParameters = sum(p.numel() for p in self.onlineNetwork.parameters())
                print(f"Model has {numParameters} parameters")
                for name, param in self.onlineNetwork.named_parameters():
                    print(f"{name}: {param.numel()} parameters")


            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No model found to load")