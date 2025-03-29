from agent import DiscreteEpsilonGreedyAgent
from gymnasium.wrappers import RecordVideo
from tetris_gymnasium.envs import Tetris
import gymnasium as gym
import secrets
import sys
import Utils

debugParameters = { 
    #"renderMode": "ansi",
    #"timeStepDelaySecs": 0.5,
    "numEpisodes": 100,
    "numTotalSteps": 100_000,
    "plotRollingLength": 10
}

hyperParameters = {
    "epsilon": 0.05,
}

def main():
    if (len(sys.argv) < 2):
        print("No model path provided")
        return
    
    seed = secrets.randbits(32)     # numpy seed needs to be between 0 and 2^32 - 1
    modelPath = sys.argv[1]
    
    renderMode = debugParameters.get("renderMode", None)
    env = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=renderMode)
    env.reset(seed=seed)
    
    agent = DiscreteEpsilonGreedyAgent(
        seed=seed,
        numActions=env.action_space.n,
        randomActionFn=lambda: env.action_space.sample(),
        modelPath=modelPath,
        train=False,
        hyperParameters=hyperParameters)

    totalStepsList, totalRewardList = Utils.runBatchEpisodes(
                                        env, 
                                        agent, 
                                        debugParameters["numEpisodes"],
                                        debugParameters["numTotalSteps"],
                                        train=False,
                                        renderMode=renderMode,
                                        timeStepDelay=debugParameters.get("timeStepDelaySecs", 0))
    env.close()

    # Visualization
    pairs = [("Episode lengths", totalStepsList), ("Episode rewards",totalRewardList)]
    Utils.plotBatchResults(pairs, debugParameters["plotRollingLength"])

if __name__ == "__main__":
    main()