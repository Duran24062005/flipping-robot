# PRD: Rediseño del humanoide articulado

## Problema y objetivo

El modelo MJCF actual usa extremidades incompletas, torso rígido y pies rectangulares. Se rediseñará como un humanoide ligero, simétrico y reconocible, con formas redondeadas y juntas suficientes para controlar torso, brazos y piernas.

## Alcance y actores

- Usuarios del entrenamiento PPO y de la reproducción visual MuJoCo.
- Reemplazar la geometría y topología física del MJCF manteniendo el nombre y orden de la raíz libre `root`.
- Añadir cuello, cintura, pecho, hombros, codos, muñecas, caderas en tres ejes, rodillas y tobillos en dos ejes.
- Mantener el entorno C++ y las APIs Python compatibles con tamaños de observación/acción leídos dinámicamente del modelo.

## Enfoque y contratos

- Geometría primaria con cápsulas, elipsoides y esferas; masas explícitas y parámetros comunes de fricción/amortiguación.
- Masa total objetivo de 19,65 kg, raíz pélvica a escala humana simplificada y pies elipsoidales que apoyan sobre el suelo en el reset.
- Veintisiete motores con límites diferenciados por región; el límite global de seguridad de C++ y el actor permanece en ±45 Nm.
- El primer `freejoint` sigue llamándose `root`; C++ usa sus posiciones/velocidades para reset, altura, orientación y velocidad de rotación.
- La pose inicial vive en `qpos0` del MJCF para que el XML sea la fuente única de verdad. El checkpoint nuevo usa `checkpoints/robot-humanoid-v2.pt`.
- La política antigua no es compatible con las nuevas dimensiones. `play.py` la rechaza antes de abrir la GUI y recomienda entrenar un checkpoint nuevo.

## Validación y riesgos

- Cargar MJCF con MuJoCo y comprobar nombres, tipos, límites, masas y dimensiones esperadas.
- Confirmar que las posiciones iniciales de ambos pies están junto al suelo y que la simulación permanece numéricamente finita durante el paso inicial.
- Construir el módulo C++, probar reset/paso del entorno y comprobar la compatibilidad de la interfaz visual.
- Un modelo más articulado aumenta el espacio de exploración; los resultados de entrenamiento previos no se transfieren y PPO puede necesitar más pasos o recompensa curricular.
