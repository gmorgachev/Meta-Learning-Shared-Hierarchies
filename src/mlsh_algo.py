import torch
import wandb
import numpy as np

from . import utils


def warmup(a2c, pool, n_iter, step_size, n_steps, n_env):
    """
    Warmup period for master policy train
    :param a2c: optimizer
    :param pool: pool of environments
    :param n_iter: number of optimizer step
    :param step_size: master policy step size
    :param n_steps: number of master step per iteration
    :param n_env: number of environments in batch
    """
    for _ in range(n_iter):
        obs_seq, _, idxs, rewards, _, _, _, m_logits, m_values = \
            pool.master_interact(n_steps, step_size)
        loss, grad_norm, entropy, values, al, cl = a2c.step(
            idxs,
            rewards.sum(-1),
            np.ones((n_env, n_steps)),
            m_logits,
            m_values
        )


def joint_train(m_a2c, s_a2c, pool, n_iter, step_size, n_steps, n_env, verbose=False):
    """
    Joint train of master policy and subpolicies
    :param m_a2c: optimizer for master policy
    :param s_a2c: optimizer for subpolicies
    :param pool: pool of environments
    :param n_iter: number of optimizer step
    :param step_size: master policy step size
    :param n_steps: number of master step per iteration
    :param n_env: number of environments in batch
    :param verbose: log per iter reward or not
    :return:
    """
    rewards_per_epoch = 0
    for i in range(n_iter):
        _, act_seq, idxs, rewards, is_alive_seq, logits, values, m_logits, m_values =\
            pool.master_interact(n_steps, step_size)
        loss, grad_norm, entropy, _, al, cl = s_a2c.step(
            act_seq,
            np.concatenate(rewards.swapaxes(0, -1), 0).swapaxes(0, -1),
            is_alive_seq,
            logits,
            values)

        m_a2c.step(
            idxs,
            rewards.sum(-1),
            np.ones((n_env, n_steps)),
            m_logits,
            m_values
        )
        rewards_per_epoch += rewards.mean()
        if verbose:
            wandb.log({
                "rewards": np.mean(rewards),
                "policy_loss": al,
                "value_loss": cl,
                "entropy": entropy,
                "loss": loss,
                "grad_norm": grad_norm
            })

    rewards_per_epoch += rewards.mean()
    return rewards_per_epoch / n_iter, grad_norm, entropy, loss


def common_train(a2c, pool, n_iter, n_steps):
    rewards_per_epoch = 0
    for i in range(n_iter):
        _, act_seq, idxs, rewards, is_alive_seq, logits, values, _, _ =\
            pool.master_interact(1, n_steps)
        loss, grad_norm, entropy, _, al, cl = a2c.step(
            act_seq,
            np.concatenate(rewards.swapaxes(0, -1), 0).swapaxes(0, -1),
            is_alive_seq,
            logits,
            values)

        rewards_per_epoch += rewards.mean()
        wandb.log({
            "rewards": np.mean(rewards.mean()),
            "policy_loss": al,
            "value_loss": cl,
            "entropy": entropy,
            "loss": loss,
            "grad_norm": grad_norm
        })

    rewards_per_epoch += rewards.mean()

    return rewards_per_epoch / n_iter, grad_norm, entropy, loss