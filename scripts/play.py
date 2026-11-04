"""Carga una política entrenada y representa su movimiento en MuJoCo.

Usa dos simulaciones sincronizadas: el entorno C++ calcula la observación que
recibe la política y una instancia Python de MuJoCo dibuja las mismas acciones.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from env import ROOT, RobotEnv
from train import ActorCritic, TORQUE_LIMIT


def parse_args() -> argparse.Namespace:
    """Valida las opciones de reproducción antes de abrir recursos gráficos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "robot.pt")
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

    # Carga la arquitectura de la red y valida sus pesos con load_state_dict.
    policy = ActorCritic(int(checkpoint["obs_size"]), int(checkpoint["action_size"]))
    try:
        policy.load_state_dict(checkpoint["model"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise SystemExit(f"Pesos incompatibles con esta arquitectura: {error}") from error
    policy.eval()

    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "humanoid.xml"))
    if model.nu != int(checkpoint["action_size"]):
        raise SystemExit("El número de actuadores no coincide con el checkpoint.")
    data = mujoco.MjData(model)
    # env y viewer mantienen el mismo estado: reset copia la observación inicial
    # al visor; en cada paso se aplican iguales controles y cantidad de subpasos.
    with RobotEnv() as env, mujoco.viewer.launch_passive(model, data) as viewer:
        frame_skip = max(1, round(env.dt / model.opt.timestep))
        if env.obs_size != int(checkpoint["obs_size"]):
            raise SystemExit("El tamaño de observación del modelo no coincide con el checkpoint.")
        for episode in range(args.episodes):
            observation = env.reset()
            mujoco.mj_resetData(model, data)
            data.qpos[:] = observation[:model.nq]
            data.qvel[:] = observation[model.nq:]
            mujoco.mj_forward(model, data)
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
                data.ctrl[:] = np.clip(action, -TORQUE_LIMIT, TORQUE_LIMIT)
                for _ in range(frame_skip):
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(env.dt)

            print(f"episodio={episode + 1} retorno={score:.1f}", flush=True)
            if not viewer.is_running():
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
