# Robot voltereta

Proyecto de simulación y aprendizaje por refuerzo para un humanoide que intenta dar una voltereta frontal y aterrizar de pie. MuJoCo calcula la física; el entorno de control está en C++; PPO y la política usan Python y PyTorch.

## Requisitos

- Linux x86_64, CMake, compilador C++17 y Python 3.11 o 3.12.
- MuJoCo instalado en el entorno Python (`mujoco`) y PyTorch (`torch`).
- GLFW/OpenGL disponibles para abrir la ventana de reproducción.

Python 3.14 está instalado en la máquina de desarrollo, pero PyTorch/MuJoCo no están presentes allí. Cree un entorno con Python 3.12 usando `uv` o `python3.12`.

## Preparación

```bash
cd ~/Desktop/robot-voltereta
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build build -j
```

Si MuJoCo no encuentra GLFW en Linux, instale los paquetes de desarrollo de GLFW y OpenGL de su distribución. La simulación y el entrenamiento no necesitan abrir una ventana; la demo sí.

## Uso

Entrenamiento corto para verificar el flujo:

```bash
python scripts/train.py --steps 20000 --checkpoint checkpoints/robot.pt
```

Entrenamiento más largo:

```bash
python scripts/train.py --steps 2000000 --checkpoint checkpoints/robot.pt
```

Reproducir la política guardada:

```bash
python scripts/play.py --checkpoint checkpoints/robot.pt
```

El archivo `robot.pt` contiene pesos de política, normalización de observaciones y metadatos. PPO es estocástico durante el entrenamiento; la reproducción usa la media de la política. Una voltereta estable no está garantizada por una ejecución corta y puede requerir más pasos o ajustes de recompensa.

## Diseño

- `assets/humanoid.xml`: torso, brazos, piernas, articulaciones, suelo y motores.
- `cpp/robot_env.cpp`: simulación MuJoCo y ABI C pequeña consumida por `ctypes`.
- `scripts/env.py`: adaptador Gymnasium con reinicio, paso, recompensa y límites del episodio.
- `scripts/train.py`: PPO continuo implementado con PyTorch; guarda checkpoint y estadísticas.
- `scripts/play.py`: carga un checkpoint y muestra la simulación en una ventana MuJoCo.

El controlador observa posiciones y velocidades generalizadas (incluyendo la pose libre del torso) y aplica torques a ocho motores. La recompensa favorece rotación frontal, mantener altura y terminar cerca de una rotación completa con el torso erguido. Un contacto o una caída temprana termina el episodio.
