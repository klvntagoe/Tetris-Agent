from agent import DiscreteEpsilonGreedyAgent
from matplotlib import pyplot as plt
from tetris_gymnasium.envs import Tetris
from typing import Callable
import gymnasium as gym
import numpy as np
import secrets
import sys
import time
import Utils

debugParameters = { 
    # "renderMode": "human",
    "renderMode": None,
    "numEpisodes": 100_000,
    "numTotalSteps": 10_000_000,    # > epsilonDecaySteps + learningStartPoint
    "plotRollingLength": 1000,
}

hyperParameters = {
    "epsilonStart": 1,
    "epsilonEnd": 0.1,
    "epsilonDecaySteps": 4_000_000,
    "learningRate": 0.001,
    "discountFactor": 0.99,
    "replayBufferCapacity": 1_000_000,
    "batchTransitionSampleSize": 512,
    "trainingFrequency": 4,
    "targetNetworkUpdateFrequency": 10_000,
    "checkpointRate": 100_000,
    "learningStartPoint": 1_000_000,
}

def main():
    seed = secrets.randbits(32)     # numpy seed needs to be between 0 and 2**32 - 1
    modelPath = None
    if (len(sys.argv) > 1):
        modelPath = sys.argv[1]

    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    env.reset(seed=seed)

    # Randomly sample from set of legal actions
    randomActionFn: Callable[[], int] = lambda: env.action_space.sample()
    agent = DiscreteEpsilonGreedyAgent(
        seed=seed,
        numActions=env.action_space.n,
        randomActionFn=randomActionFn,
        train=True,
        modelPath=modelPath,
        hyperParameters=hyperParameters)

    totalStepsList, totalRewardList = Utils.runBatchEpisodes(
                                        env, 
                                        agent, 
                                        debugParameters["numEpisodes"],
                                        debugParameters["numTotalSteps"],
                                        train=True)
    agent.QFunction.close()
    env.close()

    # Visualization
    pairs = [("Episode lengths", totalStepsList), ("Episode rewards",totalRewardList)]
    Utils.plotBatchResults(pairs, debugParameters["plotRollingLength"])

if __name__ == "__main__":
    main()