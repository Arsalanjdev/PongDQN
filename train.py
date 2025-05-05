import collections
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import wrappers
import dqn
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py as ale


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEAN_REWARD_BOUND = 19  # reward boundary for the last 100 episodes to stop training
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000  # maximum capacity of the buffer
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000  # How frequently sync the training model to the target model
EPSILON_DECAY_LAST_FRAME = (
    150000  # epsilon is decayed to final value after this much frames
)

EPSILON_START = 1.0
EPSILON_STOP = 0.01

# Aliases
State = np.ndarray
Action = int
BatchTensors = Tuple[
    torch.ByteTensor,  # current state
    torch.LongTensor,  # Actions
    torch.Tensor,  # rewards
    torch.BoolTensor,  # done or is_trunc
    torch.ByteTensor,  # next state
]


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    next_state: State


class ExperienceBuffer:
    """
    Stores the transitions obtained from the environment
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Experience:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[inx] for inx in indices]


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer) -> None:
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: Optional[np.ndarray] = None
        self._reset()

    def _reset(self) -> None:
        self.state, *_ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net: dqn.DQN, epsilon: float = 0.0) -> Optional[float]:
        """
        Plays a single step of the agent with epsilon-greedy startegy without tracking gradients.
        :param net:
        :param epsilon:
        :return: The accumulated total reward if we have reached a terminal state. None otherwise.
        """
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device).unsqueeze(0)
            q_values = net(state_v)
            _, act_v = torch.max(q_values, dim=1)
            action = int(act_v.item())

        new_state, reward, done, trunc, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, done or trunc, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if done or trunc:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def batch_to_tensors(batch: List[BatchTensors]) -> BatchTensors:
    """
    Converts a batch of transitions into tensors.
    :param batch:
    :return:
    """
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.done_trunc)
        next_states.append(exp.next_state)

    # converting to tensors
    states = torch.as_tensor(np.asarray(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.BoolTensor(dones).to(device)
    next_states = torch.as_tensor(np.asarray(next_states)).to(device)
    return states, actions, rewards, dones, next_states


def calc_loss(
    batch: List[Experience], training_net: dqn.DQN, target_net: dqn.DQN
) -> torch.Tensor:
    states, actions, rewards, dones, next_states = batch_to_tensors(batch)
    state_action_values = (
        training_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    )
    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

    predicted_value = next_state_values * GAMMA + rewards
    return nn.MSELoss()(state_action_values, predicted_value)


if __name__ == "__main__":
    env = wrappers.make_env("PongNoFrameskip-v4")
    training_network = dqn.DQN(env.observation_space.shape, env.action_space.n).to(
        device
    )
    target_network = dqn.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="dqn-ping-pong")
    print(training_network)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    total_rewards = []
    optimizer = torch.optim.Adam(training_network.parameters(), lr=LEARNING_RATE)
    frame_index = 0  # frame counter
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_index += 1
        epsilon = max(
            EPSILON_STOP, EPSILON_START - frame_index / EPSILON_DECAY_LAST_FRAME
        )
        reward = agent.play_step(training_network, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_index - ts_frame) / (time.time() - ts)
            ts_frame = frame_index
            ts = time.time()
            m_reward = np.mean(
                total_rewards[-100:]
            )  # mean value reward for the last 100 episodes
            print(
                f"{frame_index}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                f"eps {epsilon:.2f}, speed {speed:.2f} f/s"
            )
            writer.add_scalar("epsilon", epsilon, frame_index)
            writer.add_scalar("speed", speed, frame_index)
            writer.add_scalar("reward_100", m_reward, frame_index)
            writer.add_scalar("reward", reward, frame_index)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(
                    training_network.state_dict(), f"-params-best-{m_reward}.dat"
                )
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_index)
                break
        if len(buffer) < REPLAY_SIZE:
            continue
        if frame_index % SYNC_TARGET_FRAMES == 0:
            target_network.load_state_dict(training_network.state_dict())

        training_network.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, training_network, target_network)
        loss.backward()
        optimizer.step()
    writer.close()
