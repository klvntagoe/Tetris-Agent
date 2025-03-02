from abc import ABC, abstractmethod
import numpy as np
import random as rand


class ValueApproximator(ABC):
    
    @abstractmethod
    def evaluate(self, state):
        """ Evaluate the value of a state over entire action space """
    
    @abstractmethod
    def updateValue(self, state, reward, nextState):
        """ Evaluate the value of a state """
    @abstractmethod
    def updateActionValue(self, state, action, reward, nextState):
        """ Evaluate the value of a state action pair """

class ValueNetwork(ValueApproximator):
    randomVar = None
    def __init__(
            self, 
            seed : int,
            numActions):
        self.randomVar = rand.Random(seed)
        self.numActions = numActions

    def evaluate(self, state):
        qValues = np.random.rand(self.numActions)
        return qValues

    def updateValue(self, state, reward, nextState):
        pass

    def updateActionValue(self, state, action, reward, nextState):
        pass