"""Entrena la política de control continuo usando PPO y PyTorch.

Conecta ``RobotEnv`` (Python/ctypes -> C++/MuJoCo) con la red ActorCritic.
Cada iteración recoge transiciones, calcula GAE, actualiza actor y crítico,
y sustituye el checkpoint de forma atómica para conservar el último modelo.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from config import TORQUE_LIMIT
from env import ROOT, RobotEnv


class ActorCritic(nn.Module):
    """Red actor-crítico: produce torques y estima el retorno futuro.

    El actor muestrea una Normal en espacio latente; tanh limita la acción a
    ±TORQUE_LIMIT. Se conserva la muestra latente en el rollout para que PPO
    evalúe exactamente la misma densidad que usó al tomar cada acción.
    """

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()

        def mlp(output_size: int) -> nn.Sequential:
            """Red MLP compacta compartida en forma, no en pesos."""
            return nn.Sequential(
                nn.Linear(obs_size, 128), nn.Tanh(),
                nn.Linear(128, 128), nn.Tanh(),
                nn.Linear(128, output_size),
            )

        self.actor = mlp(action_size)
        self.critic = mlp(1)
        # Desviación inicial para explorar torques de magnitud útil.
        self.log_std = nn.Parameter(torch.ones(action_size))

    def distribution(self, observations: torch.Tensor) -> Normal:
        """Distribución latente con desviación limitada para evitar overflow."""
        mean = self.actor(observations)
        std = self.log_std.clamp(-2.5, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        """Devuelve torques acotados, log-probabilidad latente y valor V(s)."""
        distribution = self.distribution(observations)
        latent = distribution.mean if deterministic else distribution.sample()
        torques = torch.tanh(latent) * TORQUE_LIMIT
        log_probability = distribution.log_prob(latent).sum(-1)
        value = self.critic(observations).squeeze(-1)
        return torques, log_probability, value, latent


def save_checkpoint(path: Path, payload: dict) -> None:
    """Escribe primero un temporal y reemplaza el checkpoint sin dejarlo parcial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp:
            temp_name = temp.name
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args() -> argparse.Namespace:
    """Lee opciones y rechaza configuraciones inválidas antes de inicializar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500_000,
                        help="número total de acciones simuladas (predeterminado: 500000)")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "robot.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="dispositivo torch; por ejemplo cpu o cuda")
    parser.add_argument("--render", action="store_true",
                        help="abre el visor MuJoCo mientras PPO entrena")
    parser.add_argument("--render-every", type=int, default=1,
                        help="actualiza la ventana cada N acciones (física sigue cada acción)")
    parser.add_argument("--realtime", action="store_true",
                        help="espera el tiempo de simulación entre acciones; requiere --render")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps debe ser un entero mayor que cero")
    if args.seed < 0:
        parser.error("--seed no puede ser negativo")
    if args.render_every < 1:
        parser.error("--render-every debe ser un entero mayor que cero")
    if args.realtime and not args.render:
        parser.error("--realtime requiere --render")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("se pidió CUDA, pero torch.cuda.is_available() es falso")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(f"dispositivo inválido: {error}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA no está disponible en esta instalación de PyTorch")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error(f"no existe el dispositivo CUDA {device.index}")
    return args


def main() -> int:
    """Ejecuta PPO hasta alcanzar el presupuesto de pasos solicitado."""
    args = parse_args()
    # Mantiene repetible la inicialización y el orden de minibatches para una
    # misma versión de librerías y el mismo dispositivo.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    # Parámetros estándar de PPO para un controlador continuo de tamaño pequeño.
    horizon, epochs, minibatch = 1024, 8, 256
    gamma, gae_lambda = 0.99, 0.95
    episode_returns: list[float] = []

    with RobotEnv() as env, ExitStack() as resources:
        obs_size, action_size = env.obs_size, env.action_size
        visualizer = None
        if args.render:
            # Import diferido: el modo headless no inicializa glfw ni el visor.
            from visualizer import SynchronizedViewer

            visualizer = resources.enter_context(SynchronizedViewer(
                env, render_every=args.render_every, realtime=args.realtime
            ))
        policy = ActorCritic(obs_size, action_size).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        observation = env.reset()
        if visualizer is not None:
            visualizer.reset(observation)
        total_steps = 0
        episode_return = 0.0
        episode_length = 0

        while total_steps < args.steps:
            # Buffer de una trayectoria parcial; arrays NumPy evitan mantener
            # el grafo autograd durante la fase de recolección.
            count = min(horizon, args.steps - total_steps)
            observations = np.empty((count, obs_size), dtype=np.float32)
            latent_actions = np.empty((count, action_size), dtype=np.float32)
            log_probabilities = np.empty(count, dtype=np.float32)
            rewards = np.empty(count, dtype=np.float32)
            dones = np.empty(count, dtype=np.float32)
            values = np.empty(count, dtype=np.float32)

            for index in range(count):
                observations[index] = observation
                state_tensor = torch.as_tensor(observation, device=device)
                with torch.no_grad():
                    torque, log_probability, value, latent = policy.act(state_tensor)
                latent_np = latent.cpu().numpy()
                torque_np = torque.cpu().numpy()
                if not np.isfinite(torque_np).all() or not np.isfinite(value.item()):
                    raise FloatingPointError("La política produjo una acción o valor no finito.")
                latent_actions[index] = latent_np
                log_probabilities[index] = log_probability.item()
                values[index] = value.item()

                observation, reward, ended, _info = env.step(torque_np)
                if visualizer is not None:
                    if visualizer.is_running():
                        visualizer.advance(torque_np)
                    else:
                        print("Visor cerrado; PPO continúa en modo headless.", flush=True)
                        visualizer.close()
                        visualizer = None
                if not np.isfinite(reward):
                    raise FloatingPointError("El entorno devolvió una recompensa no finita.")
                rewards[index] = reward
                dones[index] = float(ended)
                total_steps += 1
                episode_return += reward
                episode_length += 1
                if ended:
                    episode_returns.append(episode_return)
                    print(
                        f"pasos={total_steps} episodio={len(episode_returns)} "
                        f"retorno={episode_return:.1f} longitud={episode_length}",
                        flush=True,
                    )
                    observation = env.reset()
                    if visualizer is not None:
                        visualizer.reset(observation)
                    episode_return, episode_length = 0.0, 0

            # GAE calcula ventajas descontadas; done enmascara el valor del
            # episodio siguiente para no propagar retorno a través de reset.
            with torch.no_grad():
                next_value = policy.critic(
                    torch.as_tensor(observation, device=device)
                ).item()
            advantages = np.zeros(count, dtype=np.float32)
            last_advantage = 0.0
            for index in reversed(range(count)):
                nonterminal = 1.0 - dones[index]
                next_v = next_value if index == count - 1 else values[index + 1]
                delta = rewards[index] + gamma * next_v * nonterminal - values[index]
                last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
                advantages[index] = last_advantage
            value_targets = advantages + values

            # Convierte el lote a tensores una vez y baraja sus índices por época.
            obs_tensor = torch.as_tensor(observations, device=device)
            latent_tensor = torch.as_tensor(latent_actions, device=device)
            old_logp_tensor = torch.as_tensor(log_probabilities, device=device)
            advantage_tensor = torch.as_tensor(advantages, device=device)
            target_tensor = torch.as_tensor(value_targets, device=device)
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )
            indices = np.arange(count)

            for _epoch in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, count, minibatch):
                    batch = torch.as_tensor(indices[start:start + minibatch], device=device)
                    distribution = policy.distribution(obs_tensor[batch])
                    new_logp = distribution.log_prob(latent_tensor[batch]).sum(-1)
                    ratio = (new_logp - old_logp_tensor[batch]).exp()
                    unclipped = ratio * advantage_tensor[batch]
                    clipped = ratio.clamp(0.8, 1.2) * advantage_tensor[batch]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    predicted_value = policy.critic(obs_tensor[batch]).squeeze(-1)
                    value_loss = 0.5 * (predicted_value - target_tensor[batch]).square().mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError("La pérdida PPO no es finita; se detuvo el entrenamiento.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5,
                                             error_if_nonfinite=True)
                    optimizer.step()

            mean_return = float(np.mean(episode_returns[-20:])) if episode_returns else math.nan
            print(
                f"actualización pasos={total_steps}/{args.steps} "
                f"retorno_medio_20={mean_return:.2f}", flush=True,
            )
            # El checkpoint se puede abrir directamente desde play.py. El
            # reemplazo atómico evita archivos incompletos ante una interrupción.
            save_checkpoint(args.checkpoint, {
                "model": policy.state_dict(),
                "obs_size": obs_size,
                "action_size": action_size,
                "seed": args.seed,
                "steps": total_steps,
                "algorithm": "PPO",
                "episode_returns": episode_returns,
                "torque_limit": TORQUE_LIMIT,
            })

    print(f"Checkpoint guardado en {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
