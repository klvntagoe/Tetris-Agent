# https://max-we.github.io/Tetris-Gymnasium/environments/tetris/
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from tetris_gymnasium.envs import Tetris
#from tetris_gymnasium.wrappers.observation import RgbObservation

from agent import DiscreteEpsilonGreedyAgent
import time

def runSingleEpisode(
        env : Tetris, 
        agent : DiscreteEpsilonGreedyAgent,
        debugParameters : dict, 
        train : bool = True):
    
    if not train:
        renderMode = debugParameters["renderMode"]
        render_if_needed = lambda: print(env.render() + "\n") if renderMode == "ansi" else (env.render() if renderMode == "rgb_array" or "human" else None)
    
    # Start the episode
    initialObservation, info = env.reset()
    totalReward = 0
    totalSteps = 0
    totalLinesCleared = 0
    if not train:
        render_if_needed()

    # Take first step
    first_action = agent.start(initialObservation)
    observation, reward, terminated, truncated, info = env.step(first_action)
    totalSteps += 1
    totalReward += reward
    totalLinesCleared += info["lines_cleared"]
    if not train:
        render_if_needed()

    # Take remaining steps
    while not terminated:
        action = agent.step(observation, reward, train)
        observation, reward, terminated, truncated, info = env.step(action)
        totalSteps += 1
        totalReward += reward
        totalLinesCleared += info["lines_cleared"]
        if not train:
            render_if_needed()
            print(f"e: {agent.numEpisodes} - t: {totalSteps} - epsilon: {agent.epsilon} - lastAction: {action} - reward: {reward} - totalReward: {totalReward} - info: {info}")
            timeStepDelay = debugParameters["timeStepDelay"]
            time.sleep(timeStepDelay) if timeStepDelay is not None else None

    # End the episode
    agent.end(observation, reward, train)
    return (totalSteps, totalReward, totalLinesCleared)


def main():
    # Create the environment and agent
    
    debugParameters = { 
        "renderMode": "human",
        "timeStepDelay": 0.0001,
    }
    env: Tetris = gym.make(
        "tetris_gymnasium/Tetris", 
        render_mode=debugParameters["renderMode"])
    
    # Seeding
    seed = 42
    env.reset(seed=seed)

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
    agent = DiscreteEpsilonGreedyAgent(
        seed=seed,
        numActions=env.action_space.n,
        hyperParameters=hyperParameters)
    runningRange = 100_000
    totalStepsList = []
    totalRewardList = []
    ema = lambda val, avg, alpha: val if avg is None else (alpha * val) + ((1 - alpha) * avg)
    alpha = 0.001
    totalStepsAvg = totalRewardAvg = totalLinesClearedAvg = None
    totalLinesClearedAggregate = 0
    start_time = time.time()
    for episodeIndex in range(0, runningRange):
        totalSteps, totalReward, totalLinesCleared = runSingleEpisode(
                                env, 
                                agent, 
                                debugParameters)
        
        totalStepsList.append(totalSteps)
        totalRewardList.append(totalReward)
        totalStepsAvg = ema(totalSteps, totalStepsAvg, alpha)
        totalRewardAvg = ema(totalReward, totalRewardAvg, alpha)
        totalLinesClearedAvg = ema(totalLinesCleared, totalLinesCleared, alpha)
        totalLinesClearedAggregate += totalLinesCleared
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60

        print(f"Game Over!"
            + f" - e: {episodeIndex}\t"
            + f" - r: {totalReward}\t"
            + f" - r_avg: {totalRewardAvg:.2f}\t"
            + f" - T: {totalSteps}\t"
            + f" - T_avg: {totalStepsAvg:.2f}\t"
            + f" - lc: {totalLinesCleared}\t"
            + f" - lc_agg: {totalLinesClearedAggregate}\t"
            + f" - lc_avg: {totalLinesClearedAvg:.2f}\t"
            + f" - T_agent_total: {agent.numTotalSteps}\t"
            + f" - T_agent_train: {agent.numTrainingSteps}\t"
            + f" - eps: {agent.epsilon}\t"
            + f" - duration: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        # time.sleep(0.01)

    print("All episodes completed!")

    # Visualization
    rolling_length = runningRange / 100
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(totalRewardList).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(totalStepsList).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    # axs[2].set_title("Training Error")
    # training_error_moving_average = (
    #     np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    #     / rolling_length
    # )
    # axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()