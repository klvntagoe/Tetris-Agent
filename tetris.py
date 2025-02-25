import gymnasium as gym
from tetris_gymnasium.envs import Tetris

env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
env.reset(seed=42)

terminated = False
while not terminated:
    print(env.render() + "\n")
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
print("Game Over!")