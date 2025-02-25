import random

class TetrisAgent:
    def __init__(self, numActions: int):
        self.numActions = numActions

    def start(self, observation) -> int:
        return self.selectAction(observation)
    
    def step(self, observation, reward) -> int:
        self.learn(observation, reward)
        return self.selectAction(observation)

    def end(self, observation, reward):
        self.learn(observation, reward)

    def selectAction(self, observation) -> int:
        return random.randint(0, self.numActions - 1) # random sample of action space
    
    def learn(self, observation, reward):
        pass