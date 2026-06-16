cmake_minimum_required(VERSION 3.18)

project(GPU-MetaD LANGUAGES CXX CUDA)

option(USE_GAS "Enable GAS support" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures")

get_filename_component(GPUMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LIBTORCH_ROOT "${GPUMD_ROOT}/libtorch" CACHE PATH "Path to libtorch; expected next to the src directory")

find_package(CUDAToolkit REQUIRED)

set(TORCH_CONFIG "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
if(NOT EXISTS "${TORCH_CONFIG}")
  message(FATAL_ERROR
    "libtorch was not found at ${LIBTORCH_ROOT}. "
    "Please place libtorch next to the src directory, e.g. ${GPUMD_ROOT}/libtorch, "
    "or configure with -DLIBTORCH_ROOT=/path/to/libtorch."
  )
endif()
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake" FORCE)
find_package(Torch REQUIRED)

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
if(USE_GAS)
  target_compile_definitions(gpu-sampling PRIVATE USE_GAS)
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
