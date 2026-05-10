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

#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cstring>

void print_welcome_information();
Gpumd_Options parse_options(int argc, char* argv[]);
int main(int argc, char* argv[])
{
  print_welcome_information();
  print_compile_information();
  print_gpu_information();

  print_line_1();
  printf("Started running GPU-Sampling.\n");
  print_line_2();
  #ifndef USE_GAS
    cudaDeviceSynchronize();
  #else
    // torch::cuda::synchronize();
  #endif
  clock_t time_begin = clock();

  Run run;

  #ifndef USE_GAS
    cudaDeviceSynchronize();
  #else
    // torch::cuda::synchronize();
  #endif
  clock_t time_finish = clock();

  Gpumd_Options options = parse_options(argc, argv);
  Run run(options.model_filename, options.run_filename);

  double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);

  print_line_1();
  printf("Time used = %f s.\n", time_used);
  print_line_2();

  print_line_1();
  printf("Finished running GPU-Sampling.\n");
  print_line_2();

  return EXIT_SUCCESS;
}

Gpumd_Options parse_options(int argc, char* argv[])
{
  Gpumd_Options options;
  for (int n = 1; n < argc; ++n) {
    std::string option(argv[n]);
    if (option == "-m" || option == "--model") {
      if (n + 1 >= argc) {
        PRINT_INPUT_ERROR("-m/--model should be followed by an xyz filename.");
      }
      options.model_filename = argv[++n];
    } else if (option == "-i" || option == "--input") {
      if (n + 1 >= argc) {
        PRINT_INPUT_ERROR("-i/--input should be followed by a run input filename.");
      }
      options.run_filename = argv[++n];
    } else {
      std::string error = "Unknown gpumd option: ";
      error += option;
      error += ".";
      PRINT_INPUT_ERROR(error.c_str());
    }
  }
  return options;
}
void print_welcome_information(void)
{
  printf("\n");
  printf("***************************************************************\n");
  printf("*                 Welcome to use GPU-Sampling                 *\n");
  printf("*           (GPUMD-based Enhanced Sampling package)           *\n");
  printf("*                       version 5.1                           *\n");
  printf("*              This is the gpusamplng executable              *\n");
  printf("***************************************************************\n");
  printf("\n");
}
