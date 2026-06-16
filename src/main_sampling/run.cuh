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

#include "main_gpumd/run.cuh"
#include "gas-monitor.cuh"
#include <memory>
#include <string>
#include <vector>

class GSRun : public Run
{
public:
  GSRun();
  GSRun(const std::string& model_filename);
  GSRun(const std::string& model_filename, const std::string& run_filename);

protected:
  void perform_a_run() override;
  void parse_one_keyword(std::vector<std::string>& tokens) override;

private:
  std::unique_ptr<TorchMonitor> p_gasps;
  bool is_pathsampling = false;
  bool is_ffs = false;
};
