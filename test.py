from agent import DiscreteEpsilonGreedyAgent
from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import secrets
import sys
import Utils

debugParameters = { 
    # "renderMode": "ansi",
    "renderMode": None,
    "timeStepDelaySecs": 0,
    "numEpisodes": 1_000,
    "plotRollingLength": 10
}

hyperParameters = {
    "epsilon": 0,
}

def main():
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    
    seed = secrets.randbits(32)     # numpy seed needs to be between 0 and 2^32 - 1
    modelPath = None
    if (len(sys.argv) > 1):
        modelPath = sys.argv[1]

    agent = DiscreteEpsilonGreedyAgent(
        seed=seed,
        numActions=env.action_space.n,
        modelPath=modelPath,
        train=False,
        hyperParameters=hyperParameters)
    env.reset(seed=seed)

    totalStepsList, totalRewardList = Utils.runBatchEpisodes(
                                        env, 
                                        agent, 
                                        debugParameters["numEpisodes"],
                                        train=False,
                                        renderMode=debugParameters["renderMode"],
                                        timeStepDelay=debugParameters["timeStepDelaySecs"])

    # Visualization
    pairs = [("Episode lengths", totalStepsList), ("Episode rewards",totalRewardList)]
    Utils.plotBatchResults(pairs, debugParameters["plotRollingLength"])

    env.close()

if __name__ == "__main__":
    main()