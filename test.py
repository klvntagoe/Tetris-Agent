import sys
from agent import DiscreteEpsilonGreedyAgent
from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import Utils

debugParameters = { 
    "renderMode": "ansi",
    "timeStepDelaySecs": 0,
    "numEpisodes": 20,
    "plotRollingLength": 1
}
hyperParameters = {
    "epsilon": 0,
}

def main():
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    
    seed = 42
    modelPath = "model/QNN1_nn_checkpoint_20250312_1508.pth"
    # if (len(sys.argv) > 1):
    #     modelPath = sys.argv[1]

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