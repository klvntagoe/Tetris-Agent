# Forked from https://github.com/Max-We/Tetris-Gymnasium/blob/main/examples/play_interactive.py
import sys

import cv2
import gymnasium as gym

from tetris_gymnasium.envs import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset()

    # Main game loop
    terminated = False
    numSteps = 0
    totalReward = 0
    totalLinesCleared = 0
    while not terminated:
        # Render the current state of the game as text
        env.render()

        # Pick an action from user input mapped to the keyboard
        action = None
        while action is None:
            key = cv2.waitKey(1)

            if key == ord("a"):
                action = env.unwrapped.actions.move_left
            elif key == ord("d"):
                action = env.unwrapped.actions.move_right
            elif key == ord("s"):
                action = env.unwrapped.actions.move_down
            elif key == ord("w"):
                action = env.unwrapped.actions.rotate_counterclockwise
            elif key == ord("e"):
                action = env.unwrapped.actions.rotate_clockwise
            elif key == ord(" "):
                action = env.unwrapped.actions.hard_drop
            elif key == ord("q"):
                action = env.unwrapped.actions.swap
            elif key == ord("r"):
                env.reset()
                break
                 

            if (
                cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE)
                == 0
            ):
                sys.exit()

        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)
        numSteps += 1
        totalReward += reward
        totalLinesCleared += info["lines_cleared"]

        print(f"Game Over!"
            + f" - t: {numSteps}\t"
            + f" - r: {reward}\t"
            + f" - tr: {totalReward}\t"
            + f" - lc: {totalLinesCleared}\t")

    # Game over
    print("Game Over!")
    env.close()