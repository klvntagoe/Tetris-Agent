from ActionValue import ActionValueFunction
import numpy as np
import random as rand

class DiscreteEpsilonGreedyAgent:
    def __init__(
            self,
            seed: int,
            numActions: int,
            modelPath: str = None,
            train: bool = True,
            hyperParameters: dict = None):
        # Seeding
        rand.seed(seed)
        np.random.seed(seed)

        self.hyperParameters = hyperParameters
        self.NumActions = numActions

        self.QFunction = ActionValueFunction(
            seed, 
            self.NumActions, 
            modelPath=modelPath,
            train=train,
            hyperParameters=self.hyperParameters)
        
        self.epsilon = hyperParameters.get("epsilon")
        if (self.epsilon is None):
            self.epsilonDecay = True
            self.epsilonDecaySteps = hyperParameters["epsilon_decay_steps"]
            self.epsilonEnd = hyperParameters["epsilon_end"]
            self.epsilonStart = hyperParameters["epsilon_start"]
            self.epsilon = self.epsilonStart
        else:
            self.epsilonDecay = False
        
        self.lastState = None
        self.lastAction = None
        self.numTotalSteps = 0
        self.numEpisodes = 0
        self.numTrainingSteps = 0

    def start(self, observation) -> int:
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        return self.lastAction
    
    def step(self, observation, reward, shouldTrain = True) -> int:
        if shouldTrain:
            self.learn(self.lastState, self.lastAction, reward, observation)
        self.lastState = observation
        self.lastAction = self.selectAction(observation)
        return self.lastAction

    def end(self, observation, reward, shouldTrain = True):
        if shouldTrain:
            self.learn(self.lastState, self.lastAction, reward, observation)
        self.QFunction.signalEpisodeEnd()
        self.numEpisodes += 1

    def selectAction(self, state) -> int:
        action = np.argmax(self.QFunction.evaluate(state)) if rand.random() > self.epsilon else rand.randint(0, self.NumActions - 1)
        self.numTotalSteps += 1
        return action
    
    def learn(self, state, action, reward, nextState):
        self.QFunction.update(state, action, reward, nextState)
        self.numTrainingSteps += 1
        if self.epsilonDecay:
            self.epsilon = self.epsilonEnd + ((self.epsilonStart - self.epsilonEnd) * max(0, 1 -  (self.numTrainingSteps / self.epsilonDecaySteps)))