// Núcleo de simulación C++ consumido desde Python mediante ctypes.
// No se expone ningún tipo interno de MuJoCo a Python: solo punteros opacos,
// arreglos contiguos de double y códigos de estado definidos en la ABI C.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace {
// El error se conserva por hilo para que ctypes pueda recuperar el motivo de
// un fallo sin permitir que una excepción C++ cruce la frontera C.
thread_local std::string last_error;

bool finite_array(const double* values, int count) {
  return values && std::all_of(values, values + count,
                               [](double value) { return std::isfinite(value); });
}
}  // namespace

// Estado de una instancia: modelo, datos mutables y contadores del episodio.
// Cada RobotEnv posee su propio mjData; el mjModel se carga por instancia para
// mantener sencillo el ciclo de vida y evitar estado global compartido.
struct RobotEnv {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  int frame_skip = 10;
  int elapsed_steps = 0;
  double rotation = 0.0;
  double max_seconds = 5.0;

  ~RobotEnv() {
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
  }
};

extern "C" {

// Construye el modelo y su estado dinámico. Devuelve nullptr y deja el detalle
// consultable con robot_last_error si el XML o la asignación de memoria falla.
RobotEnv* robot_create(const char* xml_path) {
  last_error.clear();
  if (!xml_path || !*xml_path) {
    last_error = "La ruta del modelo MuJoCo está vacía.";
    return nullptr;
  }

  auto env = std::make_unique<RobotEnv>();
  char message[1024] = {};
  env->model = mj_loadXML(xml_path, nullptr, message, sizeof(message));
  if (!env->model) {
    last_error = message[0] ? message : "MuJoCo no pudo cargar el modelo.";
    return nullptr;
  }
  env->data = mj_makeData(env->model);
  if (!env->data) {
    last_error = "MuJoCo no pudo reservar el estado de simulación.";
    return nullptr;
  }
  if (env->model->nq < 7 || env->model->nv < 6 || env->model->nu <= 0) {
    last_error = "El modelo requiere una raíz libre y al menos un actuador.";
    return nullptr;
  }
  // El cálculo de pose/recompensa usa los índices estándar del primer freejoint.
  const int root_id = mj_name2id(env->model, mjOBJ_JOINT, "root");
  if (root_id < 0 || env->model->jnt_type[root_id] != mjJNT_FREE ||
      env->model->jnt_qposadr[root_id] != 0 || env->model->jnt_dofadr[root_id] != 0) {
    last_error = "El modelo debe definir primero una junta libre llamada 'root'.";
    return nullptr;
  }
  return env.release();
}

// Devuelve el último error de la ABI en el hilo actual; la cadena permanece
// válida hasta la siguiente llamada que cambie el error.
const char* robot_last_error() { return last_error.c_str(); }

void robot_destroy(RobotEnv* env) { delete env; }

int robot_obs_size(const RobotEnv* env) {
  return env ? env->model->nq + env->model->nv : 0;
}
int robot_action_size(const RobotEnv* env) { return env ? env->model->nu : 0; }
double robot_dt(const RobotEnv* env) {
  return env ? env->model->opt.timestep * env->frame_skip : 0.0;
}

// Restaura la pose inicial. La observación concatena qpos y qvel, en ese orden.
// Devuelve 0 al tener éxito y -1 ante argumentos inválidos.
int robot_reset(RobotEnv* env, double* observation) {
  last_error.clear();
  if (!env || !observation) {
    last_error = "robot_reset recibió una instancia o salida nula.";
    return -1;
  }
  mj_resetData(env->model, env->data);
  // mj_resetData restaura qpos0 del MJCF; el archivo de modelo es la fuente
  // única de verdad para la altura y la pose inicial del robot.
  mj_forward(env->model, env->data);
  env->elapsed_steps = 0;
  env->rotation = 0.0;
  std::copy(env->data->qpos, env->data->qpos + env->model->nq, observation);
  std::copy(env->data->qvel, env->data->qvel + env->model->nv,
            observation + env->model->nq);
  return 0;
}

// Aplica torques, avanza frame_skip pasos y calcula recompensa/fin de episodio.
// Estado: -1 error de argumentos/acción, 0 episodio activo, 1 episodio terminado.
int robot_step(RobotEnv* env, const double* action, double* observation,
               double* reward) {
  last_error.clear();
  if (!env || !action || !observation || !reward) {
    last_error = "robot_step recibió un puntero nulo.";
    return -1;
  }
  const mjModel* model = env->model;
  mjData* data = env->data;
  if (!finite_array(action, model->nu)) {
    last_error = "La acción contiene NaN o infinito.";
    return -1;
  }

  // Limita el torque a los topes físicos de los motores del modelo.
  for (int i = 0; i < model->nu; ++i) {
    data->ctrl[i] = std::clamp(action[i], -45.0, 45.0);
  }
  for (int i = 0; i < env->frame_skip; ++i) mj_step(model, data);

  const double dt = model->opt.timestep * env->frame_skip;
  env->elapsed_steps += env->frame_skip;
  // Para la raíz libre, qvel[3..5] son velocidades angulares; el eje Y
  // corresponde al giro de una voltereta alrededor del eje lateral.
  env->rotation += data->qvel[4] * dt;

  // La orientación usa el cuaternión de la raíz (w, x, y, z). Esta proyección
  // indica cuánto apunta el eje vertical local hacia arriba.
  const double height = data->qpos[2];
  const double qx = data->qpos[4];
  const double qy = data->qpos[5];
  const double upright = std::clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0);
  constexpr double pi = 3.14159265358979323846;
  const double target_rotation = 2.0 * pi;
  const double phase_error = std::abs(env->rotation - target_rotation);
  const bool landed = phase_error < 0.55 && upright > 0.55 && height > 0.72 &&
                      std::abs(data->qvel[2]) < 2.5;
  const bool finite_state = finite_array(data->qpos, model->nq) &&
                            finite_array(data->qvel, model->nv);

  if (!finite_state) {
    *reward = -10.0;
    last_error = "MuJoCo produjo un estado no finito; el episodio se termina.";
  } else {
    // Recompensa por conservar altura, girar hacia delante y aterrizar erguido.
    *reward = 0.02 + 0.05 * std::max(0.0, height - 0.45) +
              2.0 * data->qvel[4] * dt + 0.02 * (upright + 1.0) * 0.5;
    if (landed) *reward += 15.0;
  }

  const bool fallen = !finite_state || height < 0.48 || height > 2.6;
  const bool timed_out = env->elapsed_steps * model->opt.timestep >= env->max_seconds;
  const bool done = fallen || landed || timed_out;
  if (finite_state && fallen) *reward -= 5.0;
  if (finite_state && timed_out && !landed) {
    *reward += 2.0 * std::exp(-phase_error);
  }

  std::copy(data->qpos, data->qpos + model->nq, observation);
  std::copy(data->qvel, data->qvel + model->nv, observation + model->nq);
  return done ? 1 : 0;
}
}  // extern "C"
