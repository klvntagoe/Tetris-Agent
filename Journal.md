# Journal

## Impl 1
Learning method:
- Deep Q-Learning with experience replay ([paper](https://arxiv.org/abs/1312.5602))

State representation (pre-processing): Focus on learning before state enrichment
- Use board only - board[24][18]
- Remove padding  - board[20][10]
- Binarize - if filled 1 else 0
- Flatten - board[200]

Action selection:
- Behaviour policy: epsilon-greedy
    - Epsilon decays linearly from 1 to 0.01 over 100K training steps

Action-Value Function:
- Q function approximated via a neural network
- Architacture:
    - layer_in:         200 input units (1D)
    - hidden_layer_1:   128 + ReLU
    - hidden_layer_2:   128 + ReLU
    - layer_out:        8 (num discrete actions)
- Training details
    - Replay buffer
        - 100 most recent observations
        - Uniform sampling batch size of 10
    - TD Error
        - gamma set to 0.99
        - Huber Loss with delta = 1
    - AdamW optimizer
    -   Learning rate set to 0.01
- Evaluation metric: exponential weighted avg over 100 consecutive episodes

### Results
#### Last Log
"Game Over! - e: 92264    - r: 10         - r_avg: 9.49   - T: 11         - T_avg: 10.58  - T_agent_total: 1302117        - T_agent_train: 1394382        - epsilon: 0.01         - duration: 03:13:35.88"

Looking back
- Trained on 90k episodes (3+ hours)
    - optimized on every time step
- EMA reward remained within [9,11] throughout
    - Line clearing returns reward of 10+ so this implies that the agent seldom or never clearned lines 
    - Nothing was leared


## Can try later
- Environment
    - vary seed? (same seed = same game)
- Training
    - Use Double DQN
        - Decouple the (online) network that selects actions from the (target) network that evaluates actions (as seen in this [pytorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html))
        - Soft updates of target network (eg. every 1000 steps)
    - Use a more sophisticated sampling strategy during experience replay
    - Train a base model to play withot auxiliary observations (queue, holder) then train a new model starting from the base model to incorperate this new information
- State Representation
    - Use n-histories of observations
- Reward
    - Reward clipping