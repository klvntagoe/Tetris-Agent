from matplotlib import pyplot as plt
from typing import List, Tuple
import numpy as np
import time

def plotBatchResults(
        pairs: List[Tuple[str, List[int]]],
        rolling_length: int):

    numPlots = len(pairs)
    fig, axs = plt.subplots(ncols=numPlots, figsize=(12, 5))
    for i in range(0,numPlots):
        title = pairs[i][0]
        values = pairs[i][1]

        axs[i].set_title(title)
        # compute and assign a rolling average of the data to provide a smoother graph
        moving_average = (
            np.convolve(
                np.array(values).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
        )
        axs[i].plot(range(len(moving_average)), moving_average)
    plt.tight_layout()
    plt.show()

exponentialMovingAverage = lambda val, avg, alpha: val if avg is None else (alpha * val) + ((1 - alpha) * avg)

def runBatchEpisodes(
        env,
        agent,
        numTotalEpisodes: int = 1,
        numTotalSteps: int = 100,
        train: bool = True,
        renderMode = None,
        timeStepDelay = None):
    totalStepsList = []
    totalRewardList = []
    alpha = 0.001
    totalStepsAvg = totalRewardAvg = totalLinesClearedAvg = None
    totalLinesClearedAggregate = 0
    start_time = time.time()

    episodeIndex = 0
    numStepsComleted = 0
    while episodeIndex < numTotalEpisodes and numStepsComleted < numTotalSteps:  # min number of interactions between both
        totalSteps, totalReward, totalLinesCleared = runSingleEpisode(
                                                        env, 
                                                        agent,
                                                        train,
                                                        renderMode,
                                                        timeStepDelay)
        
        totalStepsList.append(totalSteps)
        totalRewardList.append(totalReward)
        totalStepsAvg = exponentialMovingAverage(totalSteps, totalStepsAvg, alpha)
        totalRewardAvg = exponentialMovingAverage(totalReward, totalRewardAvg, alpha)
        totalLinesClearedAvg = exponentialMovingAverage(totalLinesCleared, totalLinesClearedAvg, alpha)
        totalLinesClearedAggregate += totalLinesCleared
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60

        print(f"Game Over!"
            + f" - e: {episodeIndex}\t"
            + f" - r: {totalReward}\t"
            + f" - r_avg: {totalRewardAvg:.4f}\t"
            + f" - T: {totalSteps}\t"
            + f" - T_avg: {totalStepsAvg:.4f}\t"
            + f" - lc: {totalLinesCleared}\t"
            + f" - lc_agg: {totalLinesClearedAggregate}\t"
            + f" - lc_avg: {totalLinesClearedAvg:.4f}\t"
            + f" - T_agent_total: {agent.numTotalSteps}\t"
            + f" - T_model_train: {agent.QFunction.numTrainingSteps}\t"
            + f" - eps: {agent.epsilon}\t"
            + f" - duration: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        
        episodeIndex += 1
        numStepsComleted += totalSteps

    print("Batch episodes completed!")
    return (totalStepsList, totalRewardList)
    

def runSingleEpisode(
        env, 
        agent,
        train: bool = True,
        renderMode = None,
        timeStepDelay = None):
    
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
    while not terminated or truncated:
        action = agent.step(observation, reward)
        observation, reward, terminated, truncated, info = env.step(action)
        totalSteps += 1
        totalReward += reward
        totalLinesCleared += info["lines_cleared"]
        if not train and renderMode is not None:
            print(f"e: {agent.numEpisodes}"
                  + f" - t: {totalSteps}"
                  + f" - epsilon: {agent.epsilon}"
                  + f" - action: {action}"
                  + f" - reward: {reward}"
                  + f" - totalReward: {totalReward}"
                  + f" - info: {info}")
            render_if_needed()
            time.sleep(timeStepDelay) if timeStepDelay is not None else None

    # End the episode
    agent.end(observation, reward)
    return (totalSteps, totalReward, totalLinesCleared)