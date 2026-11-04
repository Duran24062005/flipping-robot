cmake_minimum_required(VERSION 3.18)
project(robot_voltereta LANGUAGES CXX)

# C++ proporciona la simulación que llama el adaptador scripts/env.py.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Usa el paquete MuJoCo del mismo Python que ejecutará scripts/train.py; así
# encabezados y biblioteca compartida quedan emparejados con la instalación.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).parent)"
  OUTPUT_VARIABLE MUJOCO_PY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MUJOCO_QUERY_RESULT)
if(NOT MUJOCO_QUERY_RESULT EQUAL 0)
  message(FATAL_ERROR "No se pudo importar mujoco con ${Python3_EXECUTABLE}. Instale requirements.txt en ese entorno.")
endif()
# Las ruedas de MuJoCo suelen distribuir la biblioteca con versión en el nombre.
file(GLOB MUJOCO_LIBRARIES "${MUJOCO_PY_DIR}/libmujoco.so*" "${MUJOCO_PY_DIR}/libmujoco.dylib")
if(NOT MUJOCO_LIBRARIES)
  message(FATAL_ERROR "No se encontró la biblioteca MuJoCo en ${MUJOCO_PY_DIR}")
endif()
list(SORT MUJOCO_LIBRARIES)
list(GET MUJOCO_LIBRARIES 0 MUJOCO_LIBRARY)
if(NOT EXISTS "${MUJOCO_PY_DIR}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "No se encontraron los encabezados C de MuJoCo en ${MUJOCO_PY_DIR}/include")
endif()

# La ABI compartida es cargada en tiempo de ejecución por ctypes.CDLL.
add_library(robot_env SHARED cpp/robot_env.cpp)
target_include_directories(robot_env PRIVATE "${MUJOCO_PY_DIR}/include")
target_link_libraries(robot_env PRIVATE "${MUJOCO_LIBRARY}")
set_target_properties(robot_env PROPERTIES BUILD_RPATH "${MUJOCO_PY_DIR}" INSTALL_RPATH "${MUJOCO_PY_DIR}")
