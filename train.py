from agent import DiscreteEpsilonGreedyAgent
from matplotlib import pyplot as plt
from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import numpy as np
import time
import Utils

debugParameters = { 
    # "renderMode": "human",
    "renderMode": None,
    "numEpisodes": 1_000_000,
    "plotRollingLength": 1000,
}

# batch_size: number of transitions sampled from the replay buffer
# gamma: discount factor
# learning_rate: learning rate of the optimizer - "AdamW"
# epsilon_start: starting value of epsilon
# epsilon_end: final value of epsilon
# epsilon_decay_steps: rate of exponential decay of epsilon, higher means a slower decay
hyperParameters = {
    "batch_size": 32,
    "buffer_capacity": 10_000,
    "gamma": 0.99,
    "learning_rate": 0.01,      # Learning rate is high because the Tetris environment is simple. Want fast training
    "epsilon_start": 1,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 1_000_000,
}

def main():
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    
    seed = 42
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