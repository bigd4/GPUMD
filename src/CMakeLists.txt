cmake_minimum_required(VERSION 3.18)

project(GPU-MetaD LANGUAGES CXX CUDA)

option(USE_GAS "Enable GAS support" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
option(ENABLE_GAS_AOTI "Enable the torch.export/AOTInductor .pt2 backend when the selected LibTorch supports it" ON)
option(BUILD_GAS_AOTI_SMOKE "Build the standalone GAS AOTI package-loader smoke executable" OFF)

get_filename_component(GPUMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(DEFAULT_LIBTORCH_ROOT "${GPUMD_ROOT}/libtorch")
set(LIBTORCH_ROOT "" CACHE PATH "Manual path to libtorch. Leave empty to use GPUMD/libtorch.")
if(LIBTORCH_ROOT)
  get_filename_component(LIBTORCH_ROOT "${LIBTORCH_ROOT}" ABSOLUTE)
else()
  set(LIBTORCH_ROOT "${DEFAULT_LIBTORCH_ROOT}")
endif()

find_package(CUDAToolkit REQUIRED)

set(TORCH_CONFIG "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
if(NOT EXISTS "${TORCH_CONFIG}")
  message(FATAL_ERROR
    "libtorch was not found at ${LIBTORCH_ROOT}. "
    "Please place libtorch at ${DEFAULT_LIBTORCH_ROOT}, "
    "or configure with -DLIBTORCH_ROOT=/path/to/libtorch."
  )
endif()
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake" FORCE)
find_package(Torch REQUIRED)
set(AOTI_PACKAGE_LOADER_HEADER
  "${LIBTORCH_ROOT}/include/torch/csrc/inductor/aoti_package/model_package_loader.h")
set(GAS_AOTI_ENABLED OFF)
if(ENABLE_GAS_AOTI AND Torch_VERSION VERSION_EQUAL "2.13.0" AND
   EXISTS "${AOTI_PACKAGE_LOADER_HEADER}")
  set(GAS_AOTI_ENABLED ON)
  message(STATUS
    "GPU-Sampling .pt2 backend enabled (LibTorch ${Torch_VERSION}, CUDA toolkit ${CUDAToolkit_VERSION}).")
elseif(ENABLE_GAS_AOTI)
  message(STATUS
    "GPU-Sampling will be built in TorchScript-only compatibility mode: "
    ".pt2 requires LibTorch 2.13.0 with the AOTI package-loader header; "
    "found LibTorch ${Torch_VERSION} at ${LIBTORCH_ROOT}.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/main_gpumd
  ${CMAKE_CURRENT_SOURCE_DIR}/minimize
  ${CMAKE_CURRENT_SOURCE_DIR}/phonon
  ${CMAKE_CURRENT_SOURCE_DIR}/integrate
  ${CMAKE_CURRENT_SOURCE_DIR}/mc
  ${CMAKE_CURRENT_SOURCE_DIR}/force
  ${CMAKE_CURRENT_SOURCE_DIR}/measure
  ${CMAKE_CURRENT_SOURCE_DIR}/model
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities
  ${CMAKE_CURRENT_SOURCE_DIR}/main_nep
  ${CMAKE_CURRENT_SOURCE_DIR}/main_sampling
  ${CMAKE_CURRENT_SOURCE_DIR}/neb
)

file(GLOB SOURCES_GPUMD
  "${CMAKE_CURRENT_SOURCE_DIR}/main_gpumd/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/minimize/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/phonon/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/integrate/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/mc/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/force/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/measure/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/utilities/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/neb/*.cu"
)

file(GLOB SOURCES_NEP
  "${CMAKE_CURRENT_SOURCE_DIR}/main_nep/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/utilities/*.cu"
)

file(GLOB SOURCES_GPUSAMP
  "${CMAKE_CURRENT_SOURCE_DIR}/main_sampling/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/minimize/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/phonon/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/integrate/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/mc/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/force/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/measure/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/utilities/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/neb/*.cu"
)

# Add GPUMD run files except the standalone entry point.
file(GLOB SOURCES_GPUMD_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/main_gpumd/*.cu")
list(FILTER SOURCES_GPUMD_EXTRA EXCLUDE REGEX "main_gpumd/(main)\\.cu$")
list(APPEND SOURCES_GPUSAMP ${SOURCES_GPUMD_EXTRA})

add_executable(gpumd ${SOURCES_GPUMD})
add_executable(nep ${SOURCES_NEP})
add_executable(gpu-sampling ${SOURCES_GPUSAMP})

target_include_directories(gpumd PRIVATE ${COMMON_INCLUDE_DIRS})
target_include_directories(nep PRIVATE ${COMMON_INCLUDE_DIRS})
target_include_directories(gpu-sampling PRIVATE
  ${COMMON_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

foreach(target gpumd nep gpu-sampling)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  if(WIN32)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
    )
  endif()
endforeach()

target_link_libraries(gpumd PRIVATE
  CUDA::cufft
  CUDA::cublas
  CUDA::cusolver
)

target_link_libraries(nep PRIVATE
  CUDA::cufft
  CUDA::cublas
  CUDA::cusolver
)

target_link_libraries(gpu-sampling PRIVATE
  CUDA::cufft
  CUDA::cublas
  CUDA::cusolver
  ${TORCH_LIBRARIES}
)

target_compile_definitions(gpu-sampling PRIVATE TORCH_USE_CUDA_DSA=ON)
if(GAS_AOTI_ENABLED)
  target_compile_definitions(gpu-sampling PRIVATE GAS_HAS_AOTI=1)
endif()
if(USE_GAS)
  target_compile_definitions(gpu-sampling PRIVATE USE_GAS)
endif()

if(BUILD_GAS_AOTI_SMOKE)
  if(NOT GAS_AOTI_ENABLED)
    message(FATAL_ERROR "BUILD_GAS_AOTI_SMOKE requires the .pt2 backend")
  endif()
  add_executable(gas-aoti-smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools/aoti-runner-smoke.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/main_sampling/gas-model-runner.cu)
  target_include_directories(gas-aoti-smoke PRIVATE
    ${COMMON_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(gas-aoti-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  target_compile_definitions(gas-aoti-smoke PRIVATE GAS_HAS_AOTI=1)
  target_link_libraries(gas-aoti-smoke PRIVATE ${TORCH_LIBRARIES})
endif()

if(NOT WIN32)
  target_link_options(gpu-sampling PRIVATE -static-libgcc -static-libstdc++)
endif()

add_custom_target(clean_all
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/clean.cmake
)

file(WRITE ${CMAKE_BINARY_DIR}/clean.cmake "
  file(GLOB_RECURSE OBJECT_FILES \"*.o\" \"*.obj\")
  list(LENGTH OBJECT_FILES OBJECT_FILES_COUNT)
  if(OBJECT_FILES_COUNT GREATER 0)
    file(REMOVE \${OBJECT_FILES})
  endif()
  file(GLOB EXECUTABLES \"gpumd\" \"nep\" \"gpu-sampling\")
  list(LENGTH EXECUTABLES EXECUTABLES_COUNT)
  if(EXECUTABLES_COUNT GREATER 0)
    file(REMOVE \${EXECUTABLES})
  endif()
")

# Optional binary trajectory support; shared with the CPU-only adapter tests.
add_subdirectory(io)
target_link_libraries(gpumd PRIVATE trajectory_io)
target_link_libraries(gpu-sampling PRIVATE trajectory_io)
