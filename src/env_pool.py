import numpy as np
import torch

from src.mlsh_model import MLSHAgent


class EnvPool(object):
    def __init__(self, agent, make_env, n_parallel_games=1):
        self.agent = agent
        self.make_env = make_env
        self.envs = [self.make_env() for _ in range(n_parallel_games)]
        self.prev_observations = [env.reset() for env in self.envs]
        self.prev_memory_states = agent.get_initial_state(n_parallel_games)
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100):
        history_log = []

        for i in range(n_steps - 1):
            new_memory_states, (logit, value) = self.agent.step(
                self.prev_memory_states, self.prev_observations)
            actions = self.agent.sample_actions((logit, value))

            new_observations, cur_rewards, is_alive, infos = zip(
                *map(lambda x, y: self.env_step(x, y, new_memory_states),
                     range(len(self.envs)), actions))

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
            np.array(tensor).swapaxes(0, 1) \
            for tensor in zip(*history_log)
        ]
        obs_seq, act_seq, reward_seq, is_alive_seq = history_log

        return obs_seq, act_seq, reward_seq, is_alive_seq

    def env_step(self, i, action, new_memory_states):
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

    @staticmethod
    def discount_with_dones(rewards, is_active, gamma):
        discounted_rewards = rewards.clone()
        for t in reversed(range(rewards.shape[1] - 1)):
            discounted_rewards[:, t] += gamma * discounted_rewards[:, t + 1] * is_active[:, t]
        return discounted_rewards


class MLSHPool(EnvPool):
    def __init__(self, agent: MLSHAgent, make_env, n_parallel_games=1):
        super().__init__(agent, make_env, n_parallel_games)
        self.seeds = np.random.randint(1, 30, size=len(self.envs)).tolist()

        self.counter = 0

    def update_seeds(self):
        self.seeds = np.random.randint(1, 30, size=len(self.envs)).tolist()
        for seed, env in zip(self.seeds, self.envs):
            env.seed(int(seed))

    def interact(self, n_steps, subpolicies_id):
        history_log = []
        logits = []
        state_values = []
        self.prev_memory_states = [x.detach() for x in self.prev_memory_states]

        for i in range(1, n_steps):
            self.counter += 1
            new_memory_states, (logit, value) = self.agent.step(
                subpolicies_id, self.prev_memory_states, self.prev_observations)

            actions = self.agent.sample_actions((logit, None))

            new_observations, cur_rewards, is_alive, infos = zip(
                *map(lambda x, y: self.env_step(x, y, new_memory_states),
                     range(len(self.envs)), actions))

            history_log.append(
                (self.prev_observations, actions, cur_rewards, is_alive))
            logits.append(logit)
            state_values.append(value)

            self.prev_observations = new_observations
            self.prev_memory_states = new_memory_states

        history_log = [
            np.array(tensor).swapaxes(0, 1) \
            for tensor in zip(*history_log)
        ]
        obs_seq, act_seq, reward_seq, is_alive_seq = history_log
        logits = torch.stack(logits).permute(1, 0, 2)
        state_values = torch.stack(state_values).permute(1, 0)

        return obs_seq, act_seq, reward_seq, is_alive_seq, logits, state_values

    def master_interact(self, n_master_steps=10, step_size=5):
        history_log = []
        logits = []
        values = []

        m_logits = []
        m_values = []
        self.prev_memory_states = [x.detach() for x in self.prev_memory_states]

        for i in range(n_master_steps):
            new_memory_states, (m_logit, m_value) = self.agent.step_master(
                self.prev_memory_states, self.prev_observations)
            subpolicies_id = self.agent.sample_actions((m_logit, m_value))

            obs_seq, act_seq, reward_seq, is_alive_seq, logit, value = \
                self.interact(step_size, subpolicies_id)

            history_log.append(
                (obs_seq, act_seq, subpolicies_id, reward_seq, is_alive_seq))
            logits.append(logit)
            values.append(value)
            m_logits.append(m_logit)
            m_values.append(m_value)

        history_log = [
            np.array(tensor).swapaxes(0, 1) \
            for tensor in zip(*history_log)
        ]
        obs_seq, act_seq, idxs, reward_seq, is_alive_seq = history_log
        m_logits = torch.stack(m_logits).permute(1, 0, 2)
        m_values = torch.stack(m_values).permute(1, 0)

        logits = torch.cat(logits, 1)
        values = torch.cat(values, 1)

        act_seq = np.concatenate(act_seq.swapaxes(0, -1), 0).swapaxes(0, -1)
        is_alive_seq = np.concatenate(is_alive_seq.swapaxes(0, -1), 0).swapaxes(0, -1)

        return obs_seq, act_seq, idxs, reward_seq, is_alive_seq, logits, values, m_logits, m_values

    def get_master_idxs(self):
        with torch.no_grad():
            new_memory_states, (logit, value) = self.agent.step_master(
                self.prev_memory_states, self.prev_observations)
            subpolicies_id = self.agent.sample_actions((logit, value))

            return subpolicies_id

    def env_step(self, i, action, new_memory_states):
        if not self.just_ended[i]:
            new_observation, cur_reward, is_done, info = \
                self.envs[i].step(action)
            if is_done:
                self.just_ended[i] = True

            return new_observation, cur_reward, True, info
        else:
            self.envs[i].seed(self.seeds[i])
            new_observation = self.envs[i].reset()

            initial_memory_state = self.agent.get_initial_state(
                batch_size=1)
            for m_i in range(len(new_memory_states)):
                new_memory_states[m_i][i] = initial_memory_state[m_i][0]

            self.just_ended[i] = False

            return new_observation, 0, False, {'end': True}