"""Adaptador Python y límites seguros para el entorno C++/MuJoCo.

Este módulo traduce arreglos NumPy a la ABI C de ``cpp/robot_env.cpp``.
No implementa física: entrega acciones y devuelve observaciones, recompensa
y estado terminal a ``train.py`` o ``play.py``.
"""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

# Raíz del proyecto, usada para resolver el modelo y la biblioteca por defecto
# sin depender del directorio desde el que se invoque Python.
ROOT = Path(__file__).resolve().parents[1]
_DOUBLE_PTR = ct.POINTER(ct.c_double)


class RobotEnv:
    """Una instancia de simulación, con un episodio activo a la vez.

    ``reset`` inicia o reinicia el episodio. ``step`` solo se permite después
    de reset y antes de recibir ``done=True``. ``close`` libera el estado C++.
    La observación es un vector float32 con qpos seguido por qvel; las acciones
    son los torques float32, uno por actuador.
    """

    def __init__(self, xml_path: Path | str | None = None,
                 library: Path | str | None = None):
        lib_path = Path(library) if library else ROOT / "build" / "librobot_env.so"
        model_path = Path(xml_path) if xml_path else ROOT / "assets" / "humanoid.xml"
        self.xml_path = model_path.expanduser().resolve()
        lib_path = lib_path.expanduser().resolve()
        if not lib_path.is_file():
            raise FileNotFoundError(f"No existe {lib_path}; compile el proyecto con CMake.")
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"No existe el modelo MuJoCo: {self.xml_path}")

        # Configura explícitamente las firmas C para que ctypes no trunque
        # punteros o convierta doubles a enteros al cruzar la ABI.
        self.lib = ct.CDLL(str(lib_path))
        self.lib.robot_create.argtypes = [ct.c_char_p]
        self.lib.robot_create.restype = ct.c_void_p
        self.lib.robot_last_error.argtypes = []
        self.lib.robot_last_error.restype = ct.c_char_p
        self.lib.robot_destroy.argtypes = [ct.c_void_p]
        self.lib.robot_obs_size.argtypes = [ct.c_void_p]
        self.lib.robot_obs_size.restype = ct.c_int
        self.lib.robot_action_size.argtypes = [ct.c_void_p]
        self.lib.robot_action_size.restype = ct.c_int
        self.lib.robot_dt.argtypes = [ct.c_void_p]
        self.lib.robot_dt.restype = ct.c_double
        self.lib.robot_reset.argtypes = [ct.c_void_p, _DOUBLE_PTR]
        self.lib.robot_reset.restype = ct.c_int
        self.lib.robot_step.argtypes = [ct.c_void_p, _DOUBLE_PTR, _DOUBLE_PTR,
                                       _DOUBLE_PTR]
        self.lib.robot_step.restype = ct.c_int

        self.handle = self.lib.robot_create(str(self.xml_path).encode("utf-8"))
        if not self.handle:
            raise RuntimeError(self._last_error("MuJoCo no pudo crear el entorno."))
        self.obs_size = self.lib.robot_obs_size(self.handle)
        self.action_size = self.lib.robot_action_size(self.handle)
        self.dt = self.lib.robot_dt(self.handle)
        if self.obs_size <= 0 or self.action_size <= 0 or not np.isfinite(self.dt) or self.dt <= 0:
            self.close()
            raise RuntimeError("El modelo reportó dimensiones o dt inválidos.")

        # Buffers float64 contiguos cumplen el contrato C++; el API público
        # devuelve float32 porque es el formato que consume la red de PyTorch.
        self._obs = np.zeros(self.obs_size, dtype=np.float64)
        self._action = np.zeros(self.action_size, dtype=np.float64)
        self._reward = ct.c_double()
        self._needs_reset = True

    def _last_error(self, fallback: str) -> str:
        """Obtiene el último error del hilo C++ y conserva un mensaje útil."""
        raw = self.lib.robot_last_error()
        return raw.decode("utf-8", errors="replace") if raw else fallback

    def _ensure_open(self) -> None:
        """Evita llamadas a un puntero C++ después de liberar el entorno."""
        if not self.handle:
            raise RuntimeError("El entorno ya está cerrado.")

    def _observation(self) -> np.ndarray:
        """Valida el estado nativo antes de entregarlo a la política."""
        if not np.isfinite(self._obs).all():
            raise FloatingPointError("La observación contiene NaN o infinito.")
        return self._obs.astype(np.float32, copy=True)

    def reset(self) -> np.ndarray:
        """Reinicia la pose en MuJoCo y devuelve la observación inicial."""
        self._ensure_open()
        status = self.lib.robot_reset(
            self.handle, self._obs.ctypes.data_as(_DOUBLE_PTR)
        )
        if status != 0:
            raise RuntimeError(self._last_error("Falló robot_reset."))
        self._needs_reset = False
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Aplica un vector de torques y devuelve obs, reward, done e info."""
        self._ensure_open()
        if self._needs_reset:
            raise RuntimeError("Llame reset() antes de step() y después de cada episodio.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"La acción debe tener forma ({self.action_size},); recibida {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError("La acción contiene NaN o infinito.")

        # Copia al buffer contiguo que C++ espera; el código C++ vuelve a
        # limitar el torque a ±45 Nm como segunda barrera.
        self._action[:] = action
        status = self.lib.robot_step(
            self.handle,
            self._action.ctypes.data_as(_DOUBLE_PTR),
            self._obs.ctypes.data_as(_DOUBLE_PTR),
            ct.byref(self._reward),
        )
        if status < 0:
            raise RuntimeError(self._last_error("Falló robot_step."))
        if not np.isfinite(self._reward.value):
            raise FloatingPointError("La recompensa no es finita.")
        done = status == 1
        if status not in (0, 1):
            raise RuntimeError(f"Estado desconocido devuelto por C++: {status}.")
        observation = self._observation()
        self._needs_reset = done
        info = {"sim_time_step": self.dt}
        if done:
            info["termination_reason"] = self._last_error("episodio completado o límite de tiempo")
        return observation, float(self._reward.value), done, info

    def close(self) -> None:
        """Libera mjData y mjModel; es seguro llamarlo más de una vez."""
        if getattr(self, "handle", None):
            self.lib.robot_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "RobotEnv":
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
