import gym
import numpy as np

from gym import Wrapper


def evaluate(agent, env, n_games=1):
    """
    Plays an entire game start to end, returns session rewards.
    """
    game_rewards = []
    for _ in range(n_games):
        observation = env.reset()
        prev_memories = agent.get_initial_state(1)

        total_reward = 0
        while True:
            new_memories, readouts = agent.step(
                prev_memories, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])
            total_reward += reward
            prev_memories = new_memories
            if done:
                break

        game_rewards.append(total_reward)
    return game_rewards


class ObserverMinigrid(Wrapper):
    def __init__(self, env):
        super(ObserverMinigrid, self).__init__(env)
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return ([obs["image"]],
                np.asarray([obs["direction"]]))

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = ([obs["image"]],
                np.asarray([obs["direction"]]))
        return obs, r, done, info


def make_env(name="MiniGrid-Empty-5x5-v0"):
    from gym_minigrid.wrappers import ImgObsWrapper
    env = gym.make(name)
    env = ImgObsWrapper(env)
    return env

