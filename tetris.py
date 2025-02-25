import gymnasium as gym
from tetris_gymnasium.envs import Tetris

from agent import TetrisAgent
import time

# Create the environmenta and
env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
agent = TetrisAgent(env.action_space.n);

# Start the game
initial_observation = env.reset(seed=42)
action = agent.start(initial_observation)

# Play the game
terminated = False
while not terminated:
    print(env.render() + "\n")
    observation, reward, terminated, truncated, info = env.step(action)
    action = agent.step(observation, reward)
    time.sleep(0.1)
    if terminated:
        print(env.render() + "\n")
        agent.end(observation, reward)
        break

print("Game Over!")