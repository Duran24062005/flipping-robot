"""Visor MuJoCo sincronizado con las acciones del entorno de entrenamiento.

El entorno nativo C++ calcula las transiciones y recompensas. Esta clase posee
un segundo MjModel/MjData solo para dibujar esas mismas transiciones en la GUI.
Avanza la copia visual cada acción y limita únicamente la frecuencia de ``sync``
con la ventana; así no se omiten pasos de física al reducir la frecuencia.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from config import TORQUE_LIMIT
from env import ROOT, RobotEnv


class SynchronizedViewer:
    """Espejo visual de un ``RobotEnv`` para entrenamiento y reproducción."""

    def __init__(self, env: RobotEnv, render_every: int = 1,
                 realtime: bool = False, model_path: Path | None = None):
        if render_every < 1:
            raise ValueError("render_every debe ser mayor que cero")
        self.env = env
        self.render_every = render_every
        self.realtime = realtime
        self.model_path = model_path or ROOT / "assets" / "humanoid.xml"
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer = None
        self.frame_skip = 0
        self.steps_since_sync = 0

    def __enter__(self) -> "SynchronizedViewer":
        """Abre la ventana y valida contrato/dimensiones antes de usarla."""
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        expected_obs = self.model.nq + self.model.nv
        if self.model.nu != self.env.action_size:
            raise RuntimeError(
                f"La GUI tiene {self.model.nu} actuadores y el entorno C++ "
                f"tiene {self.env.action_size}."
            )
        if expected_obs != self.env.obs_size:
            raise RuntimeError(
                f"La GUI produciría {expected_obs} valores y el entorno C++ "
                f"produce {self.env.obs_size}."
            )
        self.frame_skip = round(self.env.dt / self.model.opt.timestep)
        if self.frame_skip < 1 or not np.isclose(
            self.frame_skip * self.model.opt.timestep, self.env.dt, rtol=1e-5, atol=1e-8
        ):
            raise RuntimeError("Los pasos de física de C++ y la GUI no coinciden.")
        self.data = mujoco.MjData(self.model)
        # En la API pasiva, Python controla el avance de física y llama sync()
        # para transferir la pose al hilo que dibuja la ventana.
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self

    def is_running(self) -> bool:
        """Indica si la ventana sigue abierta y recibe actualizaciones."""
        return self.viewer is not None and self.viewer.is_running()

    def reset(self, observation: np.ndarray) -> None:
        """Copia qpos/qvel iniciales del entorno C++ al estado visible."""
        if self.model is None or self.data is None or not self.is_running():
            return
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.model.nq + self.model.nv,):
            raise ValueError("La observación inicial no coincide con el modelo visible.")
        if not np.isfinite(observation).all():
            raise ValueError("La observación inicial contiene NaN o infinito.")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = observation[:self.model.nq]
        self.data.qvel[:] = observation[self.model.nq:]
        mujoco.mj_forward(self.model, self.data)
        self.steps_since_sync = 0
        self.viewer.sync()

    def advance(self, torque: np.ndarray) -> None:
        """Avanza la copia visual una acción y actualiza la ventana según el ritmo."""
        if not self.is_running() or self.model is None or self.data is None:
            return
        torque = np.asarray(torque, dtype=np.float64)
        if torque.shape != (self.model.nu,) or not np.isfinite(torque).all():
            raise ValueError("El vector de torques no coincide con los actuadores o no es finito.")
        self.data.ctrl[:] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps_since_sync += 1
        if self.steps_since_sync >= self.render_every:
            self.viewer.sync()
            self.steps_since_sync = 0
        if self.realtime:
            time.sleep(self.env.dt)

    def close(self) -> None:
        """Cierra el visor si sigue activo; seguro al cerrar la ventana primero."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __exit__(self, *_: object) -> None:
        self.close()
