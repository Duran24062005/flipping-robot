# PRD: Visualización del entrenamiento PPO

## Problema y objetivo

El entrenamiento PPO ejecuta MuJoCo en C++ sin ventana. Quien desarrolla no puede observar el movimiento de la política hasta guardar un checkpoint y abrir `play.py`. Se necesita una opción para observar la política mientras aprende, sin hacer obligatoria la GUI ni quitar el modo headless.

## Alcance y usuarios

- Para personas que entrenan y depuran el control del humanoide en una sesión gráfica local.
- Añade un visor opcional al flujo existente de `scripts/train.py`.
- El entrenamiento sin `--render` conserva el comportamiento headless.

## Enfoque e interfaces

- `--render` abre el visor interactivo de MuJoCo durante PPO.
- `--render-every N` actualiza la ventana cada N acciones del controlador; la simulación visual avanza todas las acciones para mantener sincronía.
- `--realtime` espera el tiempo de simulación entre acciones, facilitando observar el movimiento a velocidad real.
- Un módulo de visualización compartido sincroniza el estado inicial, los torques y los subpasos de la simulación C++ con una instancia visual MuJoCo en Python. No cambia la ABI C++ ni el formato del checkpoint.
- Si se cierra la ventana, el entrenamiento continúa en modo headless. Ctrl+C detiene el proceso según el comportamiento de consola existente.

## Reglas y casos límite

- Verificar que las dimensiones de observación, actuadores y paso temporal de ambas simulaciones sean compatibles.
- Rechazar frecuencia de render menor que uno y `--realtime` sin `--render`.
- Si no hay display o GLFW/OpenGL no está disponible, mostrar el error de apertura de GUI antes de iniciar el rollout.
- El visor representa las mismas acciones; el checkpoint se sigue actualizando al final de cada iteración PPO.

## Validación y riesgos

- Revisar que el modo headless siga arrancando sin crear un visor.
- Ejecutar un rollout breve con `--render` y comprobar reset de episodios, movimiento y guardado de checkpoint.
- Probar `--render-every` y `--realtime`; cerrar la ventana debe dejar PPO entrenando sin render.
- La segunda simulación añade costo de CPU; reducir la frecuencia de dibujo permite controlar ese costo. La similitud temporal depende de utilizar la misma versión de MuJoCo para C++ y Python.
