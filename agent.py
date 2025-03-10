from ActionValueNetwork import TetrisActionValueNetwork
import numpy as np
import random as rand

class DiscreteEpsilonGreedyAgent:
    epsilon = 0
    hyperParameters = None
    NumActions = None
    randomVar = None
    seed = None
    ValueApproximator = None

    def __init__(
            self, 
            numActions: int,
            hyperParameters: dict,
            seed: int):
        
        epsilon = hyperParameters["epsilon_start"]
        if (epsilon < 0 or epsilon > 1):
            raise ValueError("Epsilon must be between 0 and 1.")
        self.epsilon = epsilon
        self.hyperParameters = hyperParameters
        self.NumActions = numActions
        self.randomVar = rand.Random(seed)
        self.seed = seed
        self.ValueApproximator = TetrisActionValueNetwork(self.seed, self.NumActions, self.hyperParameters)

    def start(self, observation) -> int:
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        return self.lastAction
    
    def step(self, observation, reward) -> int:
        self.learn(self.lastState, self.lastAction, reward, observation)
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        return self.lastAction

    def end(self, observation, reward):
        self.learn(self.lastState, self.lastAction, reward, observation)

    def selectAction(self, state) -> int:
        qValues = self.ValueApproximator.evaluate(state)
        if self.randomVar.random() > self.epsilon:
            return np.argmax(qValues) # greedy action
        else:  
            return self.randomVar.randint(0, len(qValues) - 1) # random sample of action space
    
    def learn(self, state, action, reward, nextState):
        self.ValueApproximator.updateActionValue(state, action, reward, nextState)