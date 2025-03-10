# https://max-we.github.io/Tetris-Gymnasium/environments/tetris/
import gymnasium as gym
import tetris_gymnasium
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
    render_if_needed = lambda: print(env.render() + "\n") if renderMode == "ansi" else (env.render() if renderMode == "rgb_array" or "human" else None)
    
    # Start the episode
    initialObservation, info = env.reset(seed=seed)
    totalReward = 0
    totalSteps = 0
    render_if_needed()

    # Take first step
    first_action = agent.start(initialObservation)
    observation, reward, terminated, truncated, info = env.step(first_action)
    totalReward += reward
    totalSteps += 1
    render_if_needed()

    # Take remaining steps
    while not terminated:
        action = agent.step(observation, reward)
        observation, reward, terminated, truncated, info = env.step(action)
        totalReward += reward
        totalSteps += 1
        print(f"t: {totalSteps} - lastAction: {action} - reward: {reward} - totalReward: {totalReward}")
        render_if_needed()
        timeStepDelay = debugParameters["timeStepDelay"]
        if timeStepDelay is not None:
            time.sleep(timeStepDelay)

    # End the episode
    agent.end(observation, reward)
    return (totalReward, totalSteps)


def main():
    # Create the environment and agent
    
    debugParameters = { 
        "renderMode": "ansi",
        "timeStepDelay": 0.05,
    }
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    seed = 42

    # batch_size: number of transitions sampled from the replay buffer
    # gamma: discount factor
    # learning_rate: learning rate of the optimizer - "AdamW"
    # epsilon_start: starting value of epsilon
    # epsilon_end: final value of epsilon
    # epsilon_decay: rate of exponential decay of epsilon, higher means a slower decay
    hyperParameters = {
        "batch_size": 10,
        "gamma": 0.99,
        "learning_rate": 0.01,
        "epsilon_start": 0.1,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
    }
    agent = DiscreteEpsilonGreedyAgent(
        numActions=env.action_space.n,
        hyperParameters=hyperParameters,
        seed=seed)
    
    for episodeIndex in range(0, 1):
        totalReward, totalSteps = runSingleEpisode(
            env, 
            agent, 
            seed, 
            debugParameters)
        print(f"Game Over! - {episodeIndex} - TotalReward: {totalReward} - TotalSteps: {totalSteps}")
        time.sleep(1)
    
    print("All episodes completed!")
    env.close()

if __name__ == "__main__":
    main()