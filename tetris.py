# https://max-we.github.io/Tetris-Gymnasium/environments/tetris/
import gymnasium as gym
from tetris_gymnasium.envs import Tetris
#from tetris_gymnasium.wrappers.observation import RgbObservation

from agent import DiscreteEpsilonGreedyAgent
import time

def runSingleEpisode(
        env : Tetris, 
        agent : DiscreteEpsilonGreedyAgent,
        seed : int,
        debugParameters : dict):
    
    #env = RgbObservation(env)
    renderMode = debugParameters["renderMode"]
    render_if_needed = lambda: print(env.render() + "\n") if renderMode == "ansi" else (env.render() if renderMode == "rgb_array" else None)
    
    # Start the episode
    initial_observation = env.reset(seed=seed)
    render_if_needed()

    # Take first step
    first_action = agent.start(initial_observation)
    observation, reward, terminated, truncated, info = env.step(first_action)
    render_if_needed()

    # Take remaining steps
    while not terminated:
        action = agent.step(observation, reward)
        observation, reward, terminated, truncated, info = env.step(action)
        render_if_needed()
        timeStepDelay = debugParameters["timeStepDelay"]
        if timeStepDelay is not None:
            time.sleep(timeStepDelay)

    # End the episode
    agent.end(observation, reward)


def main():
    # Create the environment and agent
    debugParameters = { 
        "renderMode": "ansi",
        "timeStepDelay": 0.05,
    }
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    
    epsilon = 0.1
    seed = 42
    agent = DiscreteEpsilonGreedyAgent(
        numActions=env.action_space.n,
        epsilon=epsilon,
        seed=seed)
    
    for episodeIndex in range(0, 1):
        runSingleEpisode(
            env, 
            agent, 
            seed, 
            debugParameters)
        print(f"Game Over! - {episodeIndex}")
        time.sleep(1)
    
    print("All episodes completed!")
    env.close()

if __name__ == "__main__":
    main()