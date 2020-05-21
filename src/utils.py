import gym
import numpy as np

from gym import Wrapper

from src.mlsh_model import MLSHAgent
from gym_minigrid.wrappers import FlatObsWrapper


def evaluate(agent, env, n_games=1, last_env=None):
    game_rewards = []
    master_histories = []
    step_counts = []

    for _ in range(n_games):
        if last_env is not None:
            env.seed(last_env)
        observation = env.reset()
        total_reward = 0
        step_counter = 0
        master_hist = []
        master_action_idxs = [0]
        while True:
            readouts = agent.step(master_action_idxs, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])
            total_reward += reward
            master_hist.append(master_action_idxs)
            step_counter += 1
            if done:
                break

        game_rewards.append(total_reward)
        step_counts.append(step_counter)
        master_histories.append(master_hist)
    return game_rewards, step_counts, master_histories


def evaluate_mlsh(agent: MLSHAgent, env, n_games, master_step,
                  last_env=None):
    game_rewards = []
    master_histories = []
    step_counts = []

    for _ in range(n_games):
        if last_env is not None:
            env.seed(last_env)
        observation = env.reset()
        total_reward = 0
        step_counter = 0
        master_hist = []
        while True:
            if step_counter % master_step == 0:
                readouts = agent.step_master(
                    observation[None, ...])
                master_action_idxs = agent.sample_actions(readouts)
            readouts = agent.step(master_action_idxs, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])
            total_reward += reward
            master_hist.append(master_action_idxs)
            step_counter += 1
            if done:
                break

        game_rewards.append(total_reward)
        step_counts.append(step_counter)
        master_histories.append(master_hist)
    return game_rewards, step_counts, master_histories


class ObserverMinigrid(Wrapper):
    def __init__(self, env):
        super(ObserverMinigrid, self).__init__(env)
        self.env = FlatObsWrapper(env)

    def reset(self):
        obs = self.env.reset()
        return ([obs["image"]],
                np.asarray([obs["direction"]]))

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs, r, done, info


def make_env(name="MiniGrid-Empty-5x5-v0"):
    from gym_minigrid.wrappers import ImgObsWrapper
    env = gym.make(name)
    env = FlatObsWrapper(env)
    return env



