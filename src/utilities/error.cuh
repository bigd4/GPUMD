/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "gpu_macro.cuh"
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>

inline void print_gpumd_error(FILE* output, const char* kind, const char* file, int line, const char* text)
{
  fprintf(output, "%s:\n", kind);
  fprintf(output, "    File:       %s\n", file);
  fprintf(output, "    Line:       %d\n", line);
  fprintf(output, "    Error text: %s\n", text);
  fflush(output);
}

inline void print_gpumd_error(const char* kind, const char* file, int line, const char* text)
{
  print_gpumd_error(stderr, kind, file, line, text);
  print_gpumd_error(stdout, kind, file, line, text);
}

inline void print_gpumd_cuda_error(FILE* output, const char* file, int line, int code, const char* text)
{
  fprintf(output, "CUDA Error:\n");
  fprintf(output, "    File:       %s\n", file);
  fprintf(output, "    Line:       %d\n", line);
  fprintf(output, "    Error code: %d\n", code);
  fprintf(output, "    Error text: %s\n", text);
  fflush(output);
}

inline void print_gpumd_cuda_error(const char* file, int line, int code, const char* text)
{
  print_gpumd_cuda_error(stderr, file, line, code, text);
  print_gpumd_cuda_error(stdout, file, line, code, text);
}

#define CHECK(call)                                                                                \
  do {                                                                                             \
    const gpuError_t error_code = call;                                                            \
    if (error_code != gpuSuccess) {                                                                \
      print_gpumd_cuda_error(__FILE__, __LINE__, error_code, gpuGetErrorString(error_code));        \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define PRINT_SCANF_ERROR(count, n, text)                                                          \
  do {                                                                                             \
    if (count != n) {                                                                              \
      print_gpumd_error("Input Error", __FILE__, __LINE__, text);                                  \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define PRINT_INPUT_ERROR(text)                                                                    \
  do {                                                                                             \
    print_gpumd_error("Input Error", __FILE__, __LINE__, text);                                    \
    exit(1);                                                                                       \
  } while (0)

#define PRINT_KEYWORD_ERROR(keyword)                                                               \
  do {                                                                                             \
    std::string keyword_error = "'" + std::string(keyword) + "' is an invalid keyword.";            \
    print_gpumd_error("Input Error", __FILE__, __LINE__, keyword_error.c_str());                   \
    exit(1);                                                                                       \
  } while (0)

#ifdef STRONG_DEBUG
#define GPU_CHECK_KERNEL                                                                           \
  {                                                                                                \
    CHECK(gpuGetLastError());                                                                      \
    CHECK(gpuDeviceSynchronize());                                                                 \
  }
#else
#define GPU_CHECK_KERNEL                                                                           \
  {                                                                                                \
    CHECK(gpuGetLastError());                                                                      \
  }
#endif

void print_line_1(void);
void print_line_2(void);
FILE* my_fopen(const char* filename, const char* mode);
std::vector<std::string> get_tokens(const std::string& line);
std::vector<std::string> get_tokens(std::ifstream& input);
std::vector<std::string> get_tokens_without_unwanted_spaces(std::ifstream& input);
int get_int_from_token(const std::string& token, const char* filename, const int line);
double get_double_from_token(const std::string& token, const char* filename, const int line);
