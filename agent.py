from ActionValue import ActionValueFunction
import numpy as np
import random as rand

class DiscreteEpsilonGreedyAgent:
    def __init__(
            self, 
            numActions: int,
            hyperParameters: dict,
            seed: int):
        
        epsilonStart = hyperParameters["epsilon_start"]
        if (epsilonStart < 0 or epsilonStart > 1):
            raise ValueError("Epsilon start must be between 0 and 1.")
        self.epsilonStart = epsilonStart
        self.epsilon = epsilonStart

        epsilonEnd = hyperParameters["epsilon_end"]
        if (epsilonEnd < 0 or epsilonEnd > 1):
            raise ValueError("Epsilon end must be between 0 and 1.")
        self.epsilonEnd = epsilonEnd

        epsilonDecay = hyperParameters["epsilon_decay"]
        if (epsilonDecay < 1):
            raise ValueError("Epsilon decay must be between greater than 1")
        self.epsilonDecay = epsilonDecay

        self.hyperParameters = hyperParameters
        self.NumActions = numActions
        self.randomVar = rand.Random(seed)
        self.seed = seed
        self.QValueFunction = ActionValueFunction(
            self.seed, 
            self.NumActions, 
            self.hyperParameters)
        self.lastState = None
        self.lastAction = None
        self.numTotalSteps = 0
        self.numEpisodes = 0
        self.numTrainingSteps = 0

    def start(self, observation) -> int:
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        return self.lastAction
    
    def step(self, observation, reward) -> int:
        self.learn(self.lastState, self.lastAction, reward, observation)
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        self.numTotalSteps += 1
        return self.lastAction

    def end(self, observation, reward):
        self.numEpisodes += 1
        self.learn(self.lastState, self.lastAction, reward, observation)
        self.QValueFunction.signalEpisodeEnd()

    def selectAction(self, state) -> int:
        qValues = self.QValueFunction.evaluate(state)
        if self.randomVar.random() > self.epsilon:
            return np.argmax(qValues) # greedy action
        else:  
            return self.randomVar.randint(0, self.NumActions - 1) # random sample of action space
    
    def learn(self, state, action, reward, nextState):
        self.QValueFunction.update(state, action, reward, nextState)
        self.numTrainingSteps += 1
        self.epsilon = self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * max(0, 1 -  (self.numTrainingSteps / self.epsilonDecay))