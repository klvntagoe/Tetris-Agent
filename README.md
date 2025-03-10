# Trying new things
[Tetris Gymnasium](https://max-we.github.io/Tetris-Gymnasium/)

## Current Implementation

Learning method:
- Deep Q-Learning with experience replay ([paper](https://arxiv.org/abs/1312.5602))


State representation (pre-processing):
- Use board only
- Remove padding
- Binarize
- Flatten

Action-Value Function:
- DQN
- Architacture:
    - layer_in:         200 input units
    - hidden_layer_1:   128 + ReLU
    - layer_out:        8 (number of discrete actions)
- Training details
    - Replay memory buffer of 100 most recent observations
    - Huber Loss with delta = 1
    - AdamW optimizer
- Evaluation metric
     - Total rewards accumulated over an episode

Action selection:
- Behaviour policy: epsilon-greedy
    - Epsilon decays linearly from 1 to 0.01 over 100k training steps

## Can try later
- Training
    - Use Double DQN
        - Decouple the (online) network that selects actions from the (target) network that evaluates actions (as seen in this [pytorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html))
        - Soft updates of target network (eg. every 1000 steps)
    - Use a more sophisticated sampling strategy during experience replay
    - Train a base model to play withot auxiliary observations (queue, holder) then train a new model starting from the base model to incorperate this new information
- State Representation
    - Use n-histories of observations
- Reward
    - Reward clipping (eg. remove the "alife" reward)