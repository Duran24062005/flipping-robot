#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

struct RobotEnv {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  int frame_skip = 10;
  int elapsed = 0;
  double rotation = 0.0;
  double max_seconds = 5.0;
  std::string error;
  ~RobotEnv() { if (data) mj_deleteData(data); if (model) mj_deleteModel(model); }
};

extern "C" {
RobotEnv* robot_create(const char* xml_path) {
  auto env = std::make_unique<RobotEnv>();
  char message[1024] = {};
  env->model = mj_loadXML(xml_path, nullptr, message, sizeof(message));
  if (!env->model) return nullptr;
  env->data = mj_makeData(env->model);
  if (!env->data) return nullptr;
  return env.release();
}
void robot_destroy(RobotEnv* env) { delete env; }
int robot_obs_size(const RobotEnv* env) { return env ? env->model->nq + env->model->nv : 0; }
int robot_action_size(const RobotEnv* env) { return env ? env->model->nu : 0; }
double robot_dt(const RobotEnv* env) { return env ? env->model->opt.timestep * env->frame_skip : 0.0; }
void robot_reset(RobotEnv* env, double* observation) {
  mj_resetData(env->model, env->data);
  env->data->qpos[2] = 1.05;
  mj_forward(env->model, env->data);
  env->elapsed = 0;
  env->rotation = 0.0;
  std::copy(env->data->qpos, env->data->qpos + env->model->nq, observation);
  std::copy(env->data->qvel, env->data->qvel + env->model->nv, observation + env->model->nq);
}
// Returns 1 when the episode ended, otherwise 0.
int robot_step(RobotEnv* env, const double* action, double* observation, double* reward) {
  const mjModel* m = env->model;
  mjData* d = env->data;
  for (int i = 0; i < m->nu; ++i) d->ctrl[i] = std::clamp(action[i], -45.0, 45.0);
  for (int i = 0; i < env->frame_skip; ++i) mj_step(m, d);
  const double dt = m->opt.timestep * env->frame_skip;
  env->elapsed += env->frame_skip;
  // Free-joint rotational velocity is stored at qvel[3:6]; y is the sagittal flip axis.
  env->rotation += d->qvel[4] * dt;
  const double z = d->qpos[2];
  const double qw = d->qpos[3], qx = d->qpos[4], qy = d->qpos[5];
  const double upright = std::clamp(1.0 - 2.0 * (qx*qx + qy*qy), -1.0, 1.0);
  constexpr double pi = 3.14159265358979323846;
  const double target = 2.0 * pi;
  const double phase_error = std::abs(env->rotation - target);
  const bool near_landing = phase_error < 0.55 && upright > 0.55 && z > 0.72 && std::abs(d->qvel[2]) < 2.5;
  *reward = 0.02 + 0.05 * std::max(0.0, z - 0.45) + 2.0 * d->qvel[4] * dt + 0.02 * (upright + 1.0) * 0.5;
  if (near_landing) *reward += 15.0;
  bool fallen = z < 0.48 || z > 2.6 || !std::isfinite(z);
  bool timed_out = env->elapsed * m->opt.timestep >= env->max_seconds;
  bool done = fallen || near_landing || timed_out;
  if (fallen) *reward -= 5.0;
  if (timed_out && !near_landing) *reward += 2.0 * std::exp(-phase_error);
  std::copy(d->qpos, d->qpos + m->nq, observation);
  std::copy(d->qvel, d->qvel + m->nv, observation + m->nq);
  return done ? 1 : 0;
}
}
