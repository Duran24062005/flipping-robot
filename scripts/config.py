"""Valores compartidos entre el actor y las herramientas visuales."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mantener sincronizado con ctrlrange del MJCF y el límite de seguridad C++.
TORQUE_LIMIT = 45.0

# Checkpoint separado para no sobreescribir políticas del MJCF de 8 actuadores.
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "robot-humanoid-v2.pt"
