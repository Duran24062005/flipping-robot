"""Visualiza un checkpoint PPO en la ventana interactiva de MuJoCo."""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from env import ROOT, RobotEnv
from train import ActorCritic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "robot.pt")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = ActorCritic(state["obs_size"], state["action_size"])
    policy.load_state_dict(state["model"]); policy.eval()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "humanoid.xml"))
    data = mujoco.MjData(model)
    with RobotEnv() as env, mujoco.viewer.launch_passive(model, data) as viewer:
        for episode in range(args.episodes):
            obs = env.reset(); mujoco.mj_resetData(model, data)
            data.qpos[2] = 1.05; mujoco.mj_forward(model, data)
            done = False; score = 0.0
            while viewer.is_running() and not done:
                with torch.no_grad():
                    action, _, _ = policy.act(torch.as_tensor(obs), deterministic=True)
                action_np = action.numpy()
                obs, reward, done, _ = env.step(action_np); score += reward
                data.ctrl[:] = np.clip(action_np, -45, 45)
                for _ in range(10):
                    mujoco.mj_step(model, data)
                viewer.sync(); time.sleep(env.dt)
            print(f"episodio={episode + 1} retorno={score:.1f}")
            if not viewer.is_running():
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
