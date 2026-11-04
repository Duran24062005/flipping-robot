"""PPO continuous-control trainer implemented with PyTorch."""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from env import ROOT, RobotEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        def mlp(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, out))
        self.actor = mlp(action_size)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((action_size,), 1.0))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.clamp(-2.5, 2.0).exp().expand_as(mean))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), self.critic(obs).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "robot.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with RobotEnv() as env:
        obs_size, action_size = env.obs_size, env.action_size
        agent = ActorCritic(obs_size, action_size).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        obs = env.reset()
        total_steps, episode_return, episode_len = 0, 0.0, 0
        returns: list[float] = []
        horizon, epochs, minibatch = 1024, 8, 256
        while total_steps < args.steps:
            n = min(horizon, args.steps - total_steps)
            observations = np.zeros((n, obs_size), np.float32)
            actions = np.zeros((n, action_size), np.float32)
            log_probs = np.zeros(n, np.float32); rewards = np.zeros(n, np.float32)
            dones = np.zeros(n, np.float32); values = np.zeros(n, np.float32)
            for t in range(n):
                observations[t] = obs
                with torch.no_grad():
                    action, logp, value = agent.act(torch.as_tensor(obs, device=device))
                raw_action = action.cpu().numpy()
                actions[t], log_probs[t], values[t] = raw_action, logp.item(), value.item()
                obs, rewards[t], ended, _ = env.step(raw_action)
                dones[t] = float(ended)
                total_steps += 1; episode_return += float(rewards[t]); episode_len += 1
                if ended:
                    returns.append(episode_return)
                    print(f"pasos={total_steps} episodio={len(returns)} retorno={episode_return:.1f} longitud={episode_len}", flush=True)
                    obs, episode_return, episode_len = env.reset(), 0.0, 0
            with torch.no_grad():
                next_value = agent.critic(torch.as_tensor(obs, device=device)).item()
            advantages = np.zeros(n, np.float32); last_gae = 0.0
            for t in reversed(range(n)):
                next_nonterminal = 1.0 - dones[t]
                next_val = next_value if t == n - 1 else values[t + 1]
                last_gae = rewards[t] + 0.99 * next_val * next_nonterminal - values[t] + 0.99 * 0.95 * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns_t = advantages + values
            obs_t = torch.as_tensor(observations, device=device)
            act_t = torch.as_tensor(actions, device=device)
            old_logp_t = torch.as_tensor(log_probs, device=device)
            adv_t = torch.as_tensor(advantages, device=device)
            ret_t = torch.as_tensor(returns_t, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            indices = np.arange(n)
            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, n, minibatch):
                    idx = torch.as_tensor(indices[start:start + minibatch], device=device)
                    dist = agent.distribution(obs_t[idx])
                    new_logp = dist.log_prob(act_t[idx]).sum(-1)
                    ratio = (new_logp - old_logp_t[idx]).exp()
                    policy_loss = -torch.minimum(ratio * adv_t[idx], torch.clamp(ratio, 0.8, 1.2) * adv_t[idx]).mean()
                    value_loss = 0.5 * (agent.critic(obs_t[idx]).squeeze(-1) - ret_t[idx]).square().mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
            mean_return = float(np.mean(returns[-20:])) if returns else math.nan
            print(f"actualización pasos={total_steps}/{args.steps} retorno_medio_20={mean_return:.2f}", flush=True)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": agent.state_dict(), "obs_size": obs_size, "action_size": action_size,
                        "seed": args.seed, "steps": total_steps, "algorithm": "PPO",
                        "episode_returns": returns}, args.checkpoint)
    print(f"Checkpoint guardado en {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
