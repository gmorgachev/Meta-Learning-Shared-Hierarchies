import numpy as np
import torch


class EnvPool(object):
    def __init__(self, agent, make_env, n_parallel_games=1):
        self.agent = agent
        self.make_env = make_env
        self.envs = [self.make_env() for _ in range(n_parallel_games)]
        self.prev_observations = [env.reset() for env in self.envs]
        self.prev_memory_states = agent.get_initial_state(n_parallel_games)
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100):
        def env_step(i, action):
            if not self.just_ended[i]:
                new_observation, cur_reward, is_done, info = \
                    self.envs[i].step(action)
                if is_done:
                    self.just_ended[i] = True

                return new_observation, cur_reward, True, info
            else:
                new_observation = self.envs[i].reset()

                initial_memory_state = self.agent.get_initial_state(
                    batch_size=1)
                for m_i in range(len(new_memory_states)):
                    new_memory_states[m_i][i] = initial_memory_state[m_i][0]

                self.just_ended[i] = False

                return new_observation, 0, False, {'end': True}

        history_log = []

        for i in range(n_steps - 1):
            new_memory_states, (logit, value) = self.agent.step(
                self.prev_memory_states, self.prev_observations)
            actions = self.agent.sample_actions((logit, value))

            new_observations, cur_rewards, is_alive, infos = zip(
                *map(env_step, range(len(self.envs)), actions))

            history_log.append(
                (self.prev_observations, actions, cur_rewards, is_alive))

            self.prev_observations = new_observations
            self.prev_memory_states = new_memory_states

        dummy_actions = [0] * len(self.envs)
        dummy_rewards = [0] * len(self.envs)
        dummy_mask = [1] * len(self.envs)
        history_log.append(
            (self.prev_observations,
             dummy_actions,
             dummy_rewards,
             dummy_mask))

        history_log = [
            np.array(tensor).swapaxes(0, 1)\
                for tensor in zip(*history_log)
        ]
        obs_seq, act_seq, reward_seq, is_alive_seq = history_log

        return obs_seq, act_seq, reward_seq, is_alive_seq

    @staticmethod
    def discount_with_dones(rewards, is_active, gamma):
        discounted_rewards = rewards.clone()
        for t in reversed(range(rewards.shape[1]-1)):
            discounted_rewards[:, t] += gamma * discounted_rewards[:, t+1] * is_active[:, t]
        return discounted_rewards
