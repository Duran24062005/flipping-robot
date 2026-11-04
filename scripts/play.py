"""Carga una política entrenada y representa su movimiento en MuJoCo.

Usa dos simulaciones sincronizadas: el entorno C++ calcula la observación que
recibe la política y una instancia Python de MuJoCo dibuja las mismas acciones.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from env import RobotEnv
from visualizer import SynchronizedViewer
from train import ActorCritic
from config import DEFAULT_CHECKPOINT, TORQUE_LIMIT


def parse_args() -> argparse.Namespace:
    """Valida las opciones de reproducción antes de abrir recursos gráficos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes debe ser mayor que cero")
    if not args.checkpoint.is_file():
        parser.error(f"no existe el checkpoint: {args.checkpoint}")
    return args


def main() -> int:
    """Carga pesos y ejecuta episodios hasta terminar o cerrar la ventana."""
    args = parse_args()
    try:
        # El checkpoint se genera localmente con tensores y datos primitivos;
        # weights_only evita habilitar deserialización arbitraria de objetos.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"No se pudo leer {args.checkpoint}: {error}") from error
    required = {"model", "obs_size", "action_size", "torque_limit"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise SystemExit("El checkpoint no contiene el formato esperado por play.py.")
    if float(checkpoint["torque_limit"]) != TORQUE_LIMIT:
        raise SystemExit("El límite de torque del checkpoint no coincide con esta versión.")

    # Comprueba dimensiones antes de crear la GUI: un checkpoint de la antigua
    # anatomía (8 motores) no puede controlar el modelo nuevo (27 motores).
    with RobotEnv() as env:
        if env.obs_size != int(checkpoint["obs_size"]):
            raise SystemExit(
                f"Checkpoint incompatible: guarda obs_size={checkpoint['obs_size']} y el modelo "
                f"requiere {env.obs_size}. Entrene de nuevo con train.py."
            )
        if env.action_size != int(checkpoint["action_size"]):
            raise SystemExit(
                f"Checkpoint incompatible: guarda action_size={checkpoint['action_size']} y el modelo "
                f"requiere {env.action_size}. Entrene de nuevo con train.py."
            )

        # Carga la arquitectura de la red y valida los tensores guardados.
        policy = ActorCritic(env.obs_size, env.action_size)
        try:
            policy.load_state_dict(checkpoint["model"], strict=True)
        except (RuntimeError, TypeError) as error:
            raise SystemExit(f"Pesos incompatibles con esta arquitectura: {error}") from error
        policy.eval()

        # El mismo espejo visual se comparte con el entrenamiento para que ambos
        # caminos apliquen torques y subpasos con idéntica lógica.
        with SynchronizedViewer(env, realtime=True) as viewer:
            for episode in range(args.episodes):
                observation = env.reset()
                viewer.reset(observation)
                done, score = False, 0.0

                while viewer.is_running() and not done:
                    # La reproducción determinista usa la media de la política.
                    with torch.no_grad():
                        torque, _logp, _value, _latent = policy.act(
                            torch.as_tensor(observation), deterministic=True
                        )
                    action = torque.numpy()
                    observation, reward, done, _info = env.step(action)
                    score += reward
                    viewer.advance(action)

                print(f"episodio={episode + 1} retorno={score:.1f}", flush=True)
                if not viewer.is_running():
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
