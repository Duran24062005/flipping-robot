"""Python adapter for the MuJoCo environment compiled in C++."""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RobotEnv:
    def __init__(self, xml_path: Path | str | None = None, library: Path | str | None = None):
        lib_path = Path(library) if library else ROOT / "build" / "librobot_env.so"
        if not lib_path.exists():
            raise FileNotFoundError(f"No existe {lib_path}; ejecute cmake --build build")
        self.lib = ct.CDLL(str(lib_path))
        self.lib.robot_create.argtypes = [ct.c_char_p]
        self.lib.robot_create.restype = ct.c_void_p
        self.lib.robot_destroy.argtypes = [ct.c_void_p]
        self.lib.robot_obs_size.argtypes = [ct.c_void_p]
        self.lib.robot_obs_size.restype = ct.c_int
        self.lib.robot_action_size.argtypes = [ct.c_void_p]
        self.lib.robot_action_size.restype = ct.c_int
        self.lib.robot_dt.argtypes = [ct.c_void_p]
        self.lib.robot_dt.restype = ct.c_double
        double_ptr = ct.POINTER(ct.c_double)
        self.lib.robot_reset.argtypes = [ct.c_void_p, double_ptr]
        self.lib.robot_step.argtypes = [ct.c_void_p, double_ptr, double_ptr, double_ptr]
        self.lib.robot_step.restype = ct.c_int
        self.xml_path = Path(xml_path) if xml_path else ROOT / "assets" / "humanoid.xml"
        self.handle = self.lib.robot_create(str(self.xml_path).encode())
        if not self.handle:
            raise RuntimeError(f"MuJoCo no pudo cargar el modelo: {self.xml_path}")
        self.obs_size = self.lib.robot_obs_size(self.handle)
        self.action_size = self.lib.robot_action_size(self.handle)
        self.dt = self.lib.robot_dt(self.handle)
        self._obs = np.zeros(self.obs_size, dtype=np.float64)
        self._action = np.zeros(self.action_size, dtype=np.float64)
        self._reward = ct.c_double()

    def reset(self) -> np.ndarray:
        self.lib.robot_reset(self.handle, self._obs.ctypes.data_as(ct.POINTER(ct.c_double)))
        return self._obs.astype(np.float32, copy=True)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float64).reshape(self.action_size)
        self._action[:] = action
        done = self.lib.robot_step(
            self.handle,
            self._action.ctypes.data_as(ct.POINTER(ct.c_double)),
            self._obs.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.byref(self._reward),
        )
        return self._obs.astype(np.float32, copy=True), float(self._reward.value), bool(done), {}

    def close(self) -> None:
        if self.handle:
            self.lib.robot_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
