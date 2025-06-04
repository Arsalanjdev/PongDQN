# Pong-DQN
A Deep Q-Network (DQN) implementation that learns to play Pong from Atari using reinforcement learning (RL).
The current implementation features a CNN flattened into a FC network with an Agent that uses Experience Buffers.
A Target Network and a Training Network is used for stability.

The training phase captures videos and saves model parameters along with TensorBoard diagrams.



## Installation and Usage
```bash
git clone https://github.com/Arsalanjdev/PongDQN
cd PongDQN
pip install -r requirements.txt
```
Then cd into a directory of choice and run train.py inside the directory of the chosen implementation. 


## TODO
- Add support for Double DQN
- Add reward shaping options

