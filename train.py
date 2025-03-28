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
    "numEpisodes": 1_000_000,
    "numTotalSteps": 20_000_000,    #   must be greater than epsilonDecaySteps + learningStartPoint
    "plotRollingLength": 1000,
}

hyperParameters = {
    "epsilonStart": 1,
    "epsilonEnd": 0.1,
    "epsilonDecaySteps": 10_000_000,
    "learningRate": 0.001,
    "discountFactor": 0.99,
    "replayBufferCapacity": 1_000_000,
    "batchTransitionSampleSize": 32,
    "trainingFrequency": 4,
    "targetNetworkUpdateFrequency": 10_000,
    "checkpointRate": 200_000,
    "learningStartPoint": 1_000_000,
}

def main():
    if (len(sys.argv) < 2):
        print("No model path provided")
        return
    
    seed = secrets.randbits(32)     # numpy seed needs to be between 0 and 2**32 - 1
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
        hyperParameters=hyperParameters)

    totalStepsList, totalRewardList = Utils.runBatchEpisodes(
                                        env, 
                                        agent, 
                                        debugParameters["numEpisodes"],
                                        debugParameters["numTotalSteps"],
                                        train=True)


    # Visualization
    pairs = [("Episode lengths", totalStepsList), ("Episode rewards",totalRewardList)]
    Utils.plotBatchResults(pairs, debugParameters["plotRollingLength"])

    env.close()

if __name__ == "__main__":
    main()