from agent import DiscreteEpsilonGreedyAgent
from matplotlib import pyplot as plt
from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import numpy as np
import secrets
import time
import Utils

debugParameters = { 
    # "renderMode": "human",
    "renderMode": None,
    "numEpisodes": 1_000_000,
    "plotRollingLength": 1000,
}

hyperParameters = {
    "epsilon_start": 1,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 10_000_000,
    "learningRate": 0.001,
    "discountFactor": 0.99,
    "replayBufferCapacity": 1_000_000,
    "batchTransitionSampleSize": 32,
    "trainingFrequency": 4,
    "checkpointRate": 100_000,
    "learningStartPoint": 1_000_000,
}

def main():
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    
    seed = secrets.randbits(32)     # numpy seed needs to be between 0 and 2**32 - 1
    modelPath = None
    # if (len(sys.argv) > 1):
    #     modelPath = sys.argv[1]

    agent = DiscreteEpsilonGreedyAgent(
        seed=seed,
        numActions=env.action_space.n,
        train=True,
        hyperParameters=hyperParameters)
    env.reset(seed=seed)

    totalStepsList, totalRewardList = Utils.runBatchEpisodes(
                                        env, 
                                        agent, 
                                        debugParameters["numEpisodes"],
                                        train=True)


    # Visualization
    pairs = [("Episode lengths", totalStepsList), ("Episode rewards",totalRewardList)]
    Utils.plotBatchResults(pairs, debugParameters["plotRollingLength"])

    env.close()

if __name__ == "__main__":
    main()