from ActionValue import ActionValueFunction
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
import numpy as np
import random as rand

class DiscreteEpsilonGreedyAgent:
    def __init__(
            self,
            seed: int,
            numActions: int,
            randomActionFn: Callable[[], int],
            writer: SummaryWriter,
            modelPath: str = None,
            train: bool = True,
            hyperParameters:dict = None):
        # Seeding
        if seed is not None:
            rand.seed(seed)
            np.random.seed(seed)
        self.writer = writer
        self.hyperParameters = hyperParameters
        self.NumActions = numActions
        self.selectRandomAction = randomActionFn
        
        self.train = train
        if self.train:
            self.QFunction = ActionValueFunction(
                seed, 
                self.NumActions, 
                writer=writer,
                modelPath=modelPath,
                train=True,
                learningRate=hyperParameters["learningRate"],
                discountFactor =hyperParameters["discountFactor"],
                replayBufferCapacity=hyperParameters["replayBufferCapacity"],
                batchTransitionSampleSize=hyperParameters["batchTransitionSampleSize"],
                trainingFrequency=hyperParameters["trainingFrequency"],
                targetNetworkUpdateFrequency = hyperParameters['targetNetworkUpdateFrequency'],
                checkpointRate=hyperParameters["checkpointRate"])
            self.learningStartPoint = hyperParameters.get("learningStartPoint", 0)
        else:
            self.QFunction = ActionValueFunction(
                seed, 
                self.NumActions, 
                writer=writer,
                modelPath=modelPath,
                train=False)
            
        self.epsilon = self.hyperParameters.get("epsilon")
        if (self.epsilon is None):
            self.epsilonDecay = True
            self.epsilonDecaySteps = self.hyperParameters["epsilonDecaySteps"]
            self.epsilonEnd = self.hyperParameters["epsilonEnd"]
            self.epsilonStart = self.hyperParameters["epsilonStart"]
            self.epsilon = self.epsilonStart
        else:
            self.epsilonDecay = False
        
        self.lastState = None
        self.lastAction = None
        self.numTotalSteps = 0
        self.numEpisodes = 0

    def start(self, observation) -> int:
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        self.numEpisodes += 1
        return self.lastAction
    
    def step(self, observation, reward) -> int:
        if self.train:
            self.learn(self.lastState, self.lastAction, reward, observation)
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        return self.lastAction

    def end(self, observation, reward):
        if self.train:
            self.learn(self.lastState, self.lastAction, reward, None)
    
    def selectAction(self, state) -> int:
        action = np.argmax(self.QFunction.evaluate(state)) if rand.random() > self.epsilon else self.selectRandomAction()
        self.numTotalSteps += 1
        return action
    
    def learn(self, state, action, reward, nextState):
        if self.numTotalSteps >= self.learningStartPoint:
            lossValues = self.QFunction.update(state, action, reward, nextState, True)
            if lossValues is not None:
                tdError, avgQValues = lossValues
                self.logLearningLoss(tdError, avgQValues)
        else:
            _ = self.QFunction.update(state, action, reward, nextState, False)

        if self.epsilonDecay:
            numLearningSteps = max(0, self.numTotalSteps - self.learningStartPoint)     # delay epsilon decay until learning starts
            self.epsilon = self.epsilonEnd + ((self.epsilonStart - self.epsilonEnd) * max(0, 1 -  (numLearningSteps / self.epsilonDecaySteps)))
    
    def logLearningLoss(self, tdError, avgQValues):
        if self.numTotalSteps % 100 == 0:
            self.writer.add_scalar("Losses/tdError", tdError, self.numTotalSteps)
            self.writer.add_scalar("Losses/qValues_avg", avgQValues, self.numTotalSteps)