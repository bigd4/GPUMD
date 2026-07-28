# GPU-Sampling Forward Flux Sampling 完全手册

本文档描述 FFS_src 中的通用正通量采样流程。它不绑定任何具体材料体系。GaN B4-B1 只作为一个示例案例，用来说明如何选择 CV、界面和测试参数。

## 1. 目标和边界

FFS 流程用于估计从初态 A 到终态 B 的稀有事件速率，并保存一组由 A 走向 B 的动力学轨迹片段。

在本实现中：

- gpu-sampling 负责执行每一段分子动力学。
- GAS-monitor 负责在 MD 过程中读取外部 TorchScript 模型输出的 CV，并判断是否到达界面或返回失败界面。
- FFS_src/ffs_driver.py 负责调度每一段 gpu-sampling 任务、生成临时输入、管理状态池、保存成功轨迹、统计通过率和速率。
- 当前实现是单 GPU 串行调度。后续可以在 driver 层扩展成多进程或显存感知并发。

## 2. FFS 算法概念

选择一个单调区分 A 到 B 的反应坐标或集体变量 lambda。设置界面：

    lambda_0, lambda_1, ..., lambda_n

其中 lambda_0 靠近 A，lambda_n 靠近 B。正向过程定义为 lambda 增大时，direction 为 1；若反应坐标减小，direction 为 -1。

速率分解为：

    k_AB = Phi_A0 * product_i P(lambda_{i+1} | lambda_i)

含义：

- Phi_A0 是从 A 出发穿过第一界面 lambda_0 的通量。
- P(lambda_{i+1} | lambda_i) 是从界面 lambda_i 上保存的构型发射短轨迹，先到 lambda_{i+1} 而不是回到 A 的条件概率。
- 每一层界面的通过率相乘，再乘以第一界面通量，就是 A 到 B 的速率估计。

本实现保存每个成功越界片段；失败片段默认只记录次数和事件信息，不保留完整目录，除非 keep_failed 为 true。

## 3. 文件布局

推荐目录结构：

    project_root/
      src/build/gpu-sampling
      FFS_src/
        ffs_driver.py
        FFS_WORKFLOW_MANUAL.md
        ffs_config.example.yaml
        ffs-test/
          run.in
          GAScfg.yaml
          model.pt
          model.xyz
          potential files
          ffs.yaml

最小输入：

- gpu-sampling 可执行文件。
- 初态 A 的 model.xyz 或等价 restart.xyz。
- run.in 模板，包含势函数、系综、time_step、输出等 MD 设置。
- TorchScript CV 模型。
- GAScfg.yaml，包含 neighbor_rc、max_neighbors、cv_size、cv_log_interval 等 GAS monitor 基础参数。
- FFS yaml，包含界面、发射数量、成功/失败阈值、每段最大步数和输出目录。

## 4. TorchScript CV 模型约定

FFS 模式下，外部模型只需要输出 CV，不需要输出偏置力或 MetaD 信息。

GAS-monitor 会按以下优先级读取模型输出：

    cv_now
    cv
    commitor

推荐使用 cv_now，输出形状可以是标量或一维数组。若有多个 CV，用 ffs.cv_index 选择参与 FFS 判定的分量。

模型输入仍沿用 GAS/MetaD 的输入字典：

- positions
- cell
- side_array

因此同一个 CV-only 模型可以和 MetaD 时保持相同输入路径，只是输出不再需要 bias、forces、virial 等字段。

## 5. GAS-monitor 的 FFS 判定

FFS driver 会为每个任务目录生成临时文件 GAScfg.ffs.yaml，并在原 GAScfg.yaml 后追加运行时字段：

    ffs_enabled: true
    ffs_stop_on_fail: true or false
    ffs_cv_index: 0
    ffs_direction: 1
    ffs_target_interface: value
    ffs_fail_interface: value

判定规则：

- direction 为 1 时，CV 大于等于 ffs_target_interface 记为成功，状态码为 1。
- direction 为 -1 时，CV 小于等于 ffs_target_interface 记为成功，状态码为 1。
- 若 ffs_stop_on_fail 为 true，CV 回到 ffs_fail_interface 另一侧时记为失败，状态码为 -1。
- 未成功也未失败时继续 MD，状态码为 0。

GASCVlog.txt 第一列是事件状态，后续列是 CV 值。

## 6. run.in 模板规则

run.in 是 MD 设置模板，不应该手工写死 FFS 阶段逻辑。

driver 会读取模板并重新生成每个任务目录中的 run.in：

- 删除已有 FFSampling、PathSampling、MetaD、GASMD 行。
- 删除模板里的 run 行，并由 FFS yaml 决定当前阶段的 run 步数。
- 在 run 行之前插入 FFSampling model.pt GAScfg.ffs.yaml。
- 保留 potential、time_step、velocity、ensemble、dump_thermo、dump_exyz 等其他 MD 设置。
- 若 md.temperature_K 存在，覆盖 velocity 和常见 ensemble 温度。
- 若 ffs.random_seed 存在，给 velocity 行写入不同的 seed，格式为 velocity T seed N。

如果 FFS yaml 没写步数上限，driver 会 fallback 到模板 run.in 里第一条 run N。



### 6.1 速度重采样和副本独立性

FFS 的副本轨迹必须在统计上可区分。严格的动力学速率采样中，界面点最好是完整相空间点，即坐标和速度都来自前一段无偏 first-passage 轨迹。对确定性 MD，如果从同一个带速度的 restart.xyz 反复启动，且热浴随机数也相同，那么所有 replica 会完全重合，replicas_per_crossing 不再代表独立试验。

因此 driver 默认使用：

    md.resample_velocities: true

当起点 extxyz 含有 vel 或 velocity 属性时，driver 会在当前 trial 的 model.xyz 中移除速度列，使 run.in 里的 velocity 命令重新赋速。若 ffs.random_seed 存在，driver 会为每段任务生成独立 seed；若不存在，则交给 gpu-sampling 使用默认随机初始化。这等价于在界面结构上按目标温度重新热化速度，适合多副本 shooting 或流程验证；若你的目标是严格保持 first-passage 相空间分布，应把 md.resample_velocities 设为 false，并依靠更多独立 crossing states 或真实随机热浴产生分叉。

如果你明确要保留界面 crossing 的原始速度，例如使用已经包含随机热浴历史的动力学并且只发射一条连续分支，可以设置：

    md.resample_velocities: false

此时若动力学本身是确定性的，独立统计应来自更多 crossing states，而不是从同一个相点克隆多条。当前 driver 会在读取配置时把有效 replicas_per_crossing 自动压成 1。只有当积分器或热浴含真实随机噪声，且每条 trial 的噪声 seed 独立时，才适合在不重采样速度的情况下从同一相点发射多个随机分叉；这种情况可以后续再显式放开。

生产统计中应在 summary.json 和 events.jsonl 里检查 resample_velocities、velocity_seed，避免把完全重复的确定性轨迹当作多个试验。

### 6.2 全确定性动力学的采样含义

如果动力学完全确定，且不在界面重采样速度，那么随机性只能来自初态 A 的稳态采样。标准做法是：

1. 从 A 的稳态分布独立采样多个初始相空间点。
2. 运行无偏动力学，收集它们第一次穿过 lambda_0 的完整相空间点。
3. 对每个界面相空间点只发射一次，replicas_per_crossing 设为 1。
4. 若某层候选池不足或全失败，从上一层继续补充更多候选点；必要时逐层往前补，直到回到 A 稳态重新采样。

在这种严格设定下，不应把同一个坐标和速度复制成多条 trial。数值舍入或混沌敏感性也不应被当作可控随机源。



## 7. FFS yaml 字段

示例：

    run:
      executable: ../../src/build/gpu-sampling
      base_model: model.xyz
      run_template: run.in
      gas_model: CVModel.pt
      gas_config: GAScfg.yaml
      work_dir: FFS_runs
      overwrite_work_dir: false
      resume: false
      state_file: driver_state.json
      cuda_visible_devices: 0
      extra_files:
        - nep.txt

    md:
      timestep_fs: 1.0
      temperature_K: 1000
      resample_velocities: true
      cv_log_interval: 10

    ffs:
      interfaces: [0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00]
      direction: 1
      cv_index: 0
      fail_interface: 0.05
      flux_max_steps: 200000
      interface_max_steps: [20000, 20000, 30000, 30000, 50000, 80000]
      replicas_per_crossing: 8
      flux_min_crossings: 50
      flux_max_runs: 200
      min_trials_per_interface: 80
      max_trials_per_interface: 800
      min_successes_per_interface: 20
      max_successes_per_interface: 200
      keep_failed: false
      random_seed: 20260701

    adaptive:
      enabled: false
      min_interval_width: 0.02
      max_refinements: 8
      strategy: midpoint

    recovery:
      mode: stop
      max_rounds_per_interface: 0
      refill_successes: 4
      refill_max_trials: 80

### 7.1 全参数说明表

`run` 段控制文件路径、输出目录和断点续算行为。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `run.executable` | path | `src/build/gpu-sampling` | `gpu-sampling` 可执行文件路径。 |
| `run.base_model` | path | `model.xyz` | flux 阶段从 A 出发的初始结构。 |
| `run.initial_models` | list[path] | `[base_model]` | 可选初态稳态结构列表。全确定性动力学若从初态补样，建议提供多个独立相空间点。 |
| `run.run_template` | path | `run.in` | MD 输入模板。FFS driver 会重写 monitor 行和 `run N` 行。 |
| `run.gas_model` | path | `GASCVModel.pt` | TorchScript CV 模型。 |
| `run.gas_config` | path | `GAScfg.yaml` | GAS monitor 基础配置。 |
| `run.work_dir` | path | `FFS_runs` | 所有 FFS 输出的根目录。 |
| `run.overwrite_work_dir` | bool | `false` | 若 `work_dir` 已有旧 FFS 输出，是否清空后重新运行。测试样例通常设为 `true`。 |
| `run.resume` | bool | `false` | 是否从 `run.state_file` 记录的状态继续运行。续算时应保持 `overwrite_work_dir: false`。 |
| `run.state_file` | str | `driver_state.json` | 断点续算状态文件名，位于 `work_dir` 下。 |
| `run.cuda_visible_devices` | str/int/null | null | 单卡运行时使用的 GPU 编号；会写入 `CUDA_VISIBLE_DEVICES`。 |
| `run.command_prefix` | list[str] or str | `[]` | 执行命令前缀，例如 MPI/容器包装命令。当前推荐单卡串行。 |
| `run.extra_files` | list[path] | `[]` | 每个任务目录需要额外复制的文件，例如 `nep.txt`。 |

`md` 段控制每段 MD 的通用运行条件。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `md.timestep_fs` | float | `1.0` | MD 时间步，用于把步数换算成 ps。 |
| `md.temperature_K` | float/null | null | 若提供，driver 会覆盖模板中的 `velocity` 温度和常见 `ensemble temp T_start T_stop`；同时用于势垒 eV 换算。若省略，则保留模板温度。 |
| `md.resample_velocities` | bool | `true` | 是否在每次发射前移除起点 xyz 中的速度列并重新赋速。若为 `false`，当前 driver 会把有效副本数压成 1。 |
| `md.cv_log_interval` | int | GAScfg 或 `1` | 与 `GAScfg.yaml` 中的 `cv_log_interval` 对应，用于日志步数 fallback。 |

`ffs` 段控制界面、发射批次和统计停止条件。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ffs.interfaces` | list[float] | 必填 | 从 `lambda_0` 到 `lambda_n` 的界面数组，至少两个值。 |
| `ffs.direction` | int | `1` | `1` 表示 CV 增大方向，`-1` 表示 CV 减小方向。 |
| `ffs.cv_index` | int | `0` | 多 CV 输出时用于判定的 CV 分量。 |
| `ffs.fail_interface` | float | `interfaces[0]` | 判定返回 A 的失败界面。 |
| `ffs.flux_max_steps` | int | 模板第一条 `run N` | 从 A 到 `lambda_0` 的单段最大 MD 步数。 |
| `ffs.interface_max_steps` | int/list[int] | 模板第一条 `run N` | 从 `lambda_i` 到 `lambda_{i+1}` 的 trial 最大步数；可为单值或长度 `len(interfaces)-1` 的数组。 |
| `ffs.replicas_per_crossing` | int | `1` | 每个保存的界面点发射多少条子轨迹。若 `md.resample_velocities: false`，当前有效值自动为 1。 |
| `ffs.flux_min_crossings` | int | `20` | flux 阶段至少收集多少个 `lambda_0` crossing。 |
| `ffs.flux_max_runs` | int | `200` | flux 阶段最多启动多少条从 A 出发的长轨迹。 |
| `ffs.min_trials_per_interface` | int | `20` | 每个界面 trial 批次的最少轨迹数。 |
| `ffs.max_trials_per_interface` | int | `200` | 每个界面 trial 批次的最多轨迹数。若 recovery 触发额外批次，最终统计会累计所有批次。 |
| `ffs.min_successes_per_interface` | int | `10` | 每个界面批次至少收集多少个成功到达下一界面的点。 |
| `ffs.max_successes_per_interface` | int | 很大 | 每个界面批次最多保存多少个成功点。 |
| `ffs.keep_failed` | bool | `false` | 是否保留失败 trial 的完整目录；否则只记录事件并删除失败目录。 |
| `ffs.random_seed` | int/null | null | driver 层随机种子，用于起点轮换相关随机行为和 velocity seed。 |

`adaptive` 段控制自动细化界面。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `adaptive.enabled` | bool | `false` | 是否开启自适应界面细化。 |
| `adaptive.min_interval_width` | float | `0.0` | 允许自动细化的最小区间宽度。失败区间宽度小于等于该值时不再插点。 |
| `adaptive.max_refinements` | int | `0` | 单次 FFS 运行最多自动插入多少个界面。 |
| `adaptive.strategy` | str | `midpoint` | 当前支持 `midpoint`，即在失败区间中点插入新界面。 |

`recovery` 段控制失败后的补样策略和最大补充次数。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `recovery.mode` | str | `stop` | `stop` 直接停机；`previous_interface` 从上一界面补候选；`initial_state` 从初态重新补样并逐层推进。 |
| `recovery.max_rounds_per_interface` | int | `0` | 每个失败界面最多触发几轮恢复。达到上限后程序写入 `phase: failed` 并停止。 |
| `recovery.refill_successes` | int | `ffs.min_successes_per_interface` | 每轮恢复希望补充多少个新候选 state。 |
| `recovery.refill_max_trials` | int | `ffs.max_trials_per_interface` | 每轮恢复最多尝试多少条上游轨迹。 |

关键语义：`max_trials_per_interface` 限制的是当前界面的一轮 trial 批次。若一轮失败后 recovery 成功补充了候选池，driver 会再开一轮 trial 批次；最终 `interfaces.csv` 和 `summary.json` 中的 trials、successes、probability 会累计所有批次。真正防止无限补样的是 `recovery.max_rounds_per_interface`。

## 8. 运行阶段

### 8.1 Flux 阶段

目标：估计 Phi_A0，并收集第一界面 lambda_0 上的 crossing states。

driver 行为：

1. 从 run.base_model 准备 model.xyz；若 md.resample_velocities 为 true 且文件含速度列，会移除速度列并重新赋速。
2. 生成当前任务的 GAScfg.ffs.yaml。
3. 生成 run.in，其中 target interface 是 interfaces[0]，run 步数为 flux_max_steps。
4. 启动 gpu-sampling。
5. 如果 GAS-monitor 判断到达 lambda_0，保存 restart.xyz 到 states/interface_000。
6. 重复直到 crossing 数达到 flux_min_crossings，或尝试数达到 flux_max_runs。

### 8.2 Interface trial 阶段

目标：估计每一层条件概率 P(lambda_{i+1} | lambda_i)。

driver 行为：

1. 从 states/interface_i 选择起点。
2. 对每个起点发射 replicas_per_crossing 条 trial；默认每条 trial 重新赋速度并使用独立 seed。
3. target interface 设置为 interfaces[i+1]。
4. ffs_stop_on_fail 设置为 true，CV 回到 fail_interface 时提前失败停止。
5. 成功 trial 的 restart.xyz 保存到 states/interface_{i+1}。
6. 成功 trial 的完整目录复制到 success_segments/interface_{i+1}。
7. 失败 trial 默认删除目录，只在 events.jsonl 记录。
8. 到达 min_trials_per_interface 且 min_successes_per_interface 后进入下一界面。

## 9. 输出文件

work_dir 下会生成：

- states/interface_000/state_*.xyz: 第一界面 crossing states。
- states/interface_NNN/state_*.xyz: 后续界面的成功状态。
- success_segments/interface_NNN/segment_*: 成功轨迹片段完整目录。
- trials/...: 原始运行目录。keep_failed 为 false 时失败目录会删除。
- events.jsonl: 每次发射的事件记录，包括状态码、步数、CV、velocity_seed、是否重采样速度，以及 adaptive/recovery/resume 事件。
- driver_state.json: 断点续算状态文件，文件名可由 `run.state_file` 修改。
- interfaces.csv: 每个界面的 trials、successes、failures、probability。
- summary.json: 通量、条件概率乘积、速率估计、势垒估计、步数上限、速度重采样、adaptive 和 recovery 设置。

## 10. 断点续算和状态文件

`driver_state.json` 是 FFS driver 的运行状态账本。程序会在以下节点写入它：

- 每条 flux attempt 完成后，以 `phase: flux` 记录已完成 attempts、crossings、采样时间和 `states/interface_000`。
- 每条 interface trial 完成后，以 `phase: running` 记录当前批次已完成的 trials、successes/failures 和 state pool。
- 某个界面完成并推进到下一界面后。
- adaptive 插入新界面后。
- recovery 补充候选池后。
- 达到补充上限或无法继续时，以 `phase: failed` 停机。

状态文件包含：

- `phase`: 当前阶段，例如 `flux`、`running`、`interfaces`、`adaptive_refine`、`recovered`、`failed`。
- `interface_index`: 当前正在处理或失败的界面编号。
- `interfaces` 和 `interface_max_steps`: 运行时界面数组；若 adaptive 插入过界面，以这里为准。
- `state_pools`: 各界面已保存 state 的路径。
- `interface_stats` 和 `flux_stats`: 已完成和已累计的统计量。
- `adaptive_refinements`: 自动插点记录。
- `recovery_rounds`: 每个界面已经使用过的补充轮数。
- `stage_launch_counts` 和 `initial_model_cursor`: 用来避免续算时覆盖 trial 目录或重复初态轮换。
- `event_cursor`: 写状态时已经同步到 `events.jsonl` 的行数，用于崩溃恢复时做账本对账。

续算方式：

    run:
      work_dir: FFS_runs
      overwrite_work_dir: false
      resume: true
      state_file: driver_state.json

续算时 driver 会从状态文件读取动态界面和状态池，而不是重新相信 yaml 中初始 interfaces。它会先用 `event_cursor` 对账：扫描状态文件之后新增的 `events.jsonl` 事件，并检查已落盘但尚未进入状态池的 `states/interface_*` 成功结构。已经有完整事件或成功 state 的结果会被同步进 `driver_state.json`；只有半截 trial 目录、没有完整事件也没有成功 state 的计算会被视为无效，下一次会从最后一个有效结果之后重新发射。

若状态文件是 `phase: flux`，续算会先对账已完成的 flux attempt，然后只继续收集缺少的 `lambda_0` crossing；若状态文件是 `phase: running`，续算会保留已完成 trial，并只补足当前批次剩余的 trial 上限；如果对账后发现当前批次已经跑满但仍未成功，会立刻进入 adaptive/recovery/failed 判断。若状态文件已经是 `phase: failed`，直接续算不会悄悄从同一批界面态再发射轨迹，而是先检查 recovery 是否还有剩余轮数。如果你希望在失败后继续，可以提高 `recovery.max_rounds_per_interface`、增加 `recovery.refill_max_trials`，或修改界面/步数后重新开始一轮受控测试。若要完全从头跑，使用 `resume: false` 且 `overwrite_work_dir: true`，或换一个新的 `work_dir`。

## 11. 速率和势垒估计

summary.json 中的 rate_per_ps 使用：

    rate_per_ps = flux_per_ps * product_probability

其中：

    flux_per_ps = crossings / sampled_time_ps
    product_probability = product_i probability_i

势垒估计：

    barrier_kBT = -ln(product_probability)
    barrier_eV = barrier_kBT * k_B * temperature_K

注意：这个 barrier 是沿 FFS 界面概率乘积得到的有效势垒估计，不是完整平衡自由能面。

## 12. 界面选择建议

好的界面应该满足：

- lambda_0 靠近 A，但不能太靠近，否则 flux 过大且状态相关性强。
- 相邻界面的通过率不要太低。经验上单层概率在 0.05 到 0.5 之间更容易采样。
- 对高能垒固固相变，应把界面分细，并允许后期界面更长的 trial 步数。
- 若某一层概率接近 0，应该在该区间插入更多界面，或提高温度/压力/采样步数。
- 若某一层概率接近 1，界面可以适当放稀。


## 13. 自适应界面细化

自适应细化是可选探索功能，默认关闭。它解决的问题是：某个界面区间太宽，导致从 lambda_i 到 lambda_{i+1} 的 trial 在 max_trials_per_interface 内没有一次成功，于是条件概率估计为 0，流程无法继续。

触发条件：

1. adaptive.enabled 为 true。
2. 当前层达到 max_trials_per_interface 仍未达到 min_successes_per_interface。
3. 当前层 successes 为 0。若已经有少量成功但没达到 min_successes，说明主要是统计不足，driver 不自动改界面。
4. abs(lambda_{i+1} - lambda_i) 大于 adaptive.min_interval_width。
5. 总插点数没有超过 adaptive.max_refinements。
6. 对全确定性动力学，若同一上游池已耗尽，应补充上游候选 states，而不是重复克隆同一个 state。

当前策略是 midpoint：

    lambda_mid = 0.5 * (lambda_i + lambda_{i+1})

插点后，driver 会先重试 lambda_i 到 lambda_mid；成功后再继续 lambda_mid 到原来的 lambda_{i+1}。新拆出的两个区间继承原区间的 interface_max_steps。每次自动插点会写入 events.jsonl，最终 summary.json 的 adaptive.refinements 也会保留完整记录。

注意：自动细化不会降低真实能垒，只是把“跨度太大导致概率为 0”的层拆成更容易估计的条件概率。如果已经细化到最小宽度仍然过不去，driver 不应该继续无限插点。优先做法是扩充失败区间上游的界面池：从 lambda_{i-1} 继续发射以收集更多到达 lambda_i 的 first-passage states；若上游池也太窄，再继续往前补样本，必要时回到 flux 阶段重新收集 lambda_0 crossing states。若补样本和增加 interface_max_steps 后仍失败，应考虑提高试验温度/压力、调整 CV，或承认当前 CV 在该区间不是好的反应坐标。


## 14. 失败恢复策略

当某一层 lambda_i 到 lambda_{i+1} 达到 max_trials_per_interface 仍不能满足 min_successes_per_interface，且 adaptive 已关闭或不能继续细化时，driver 可以按 recovery.mode 处理：

- stop: 默认行为，直接停止并保留 events.jsonl 证据。
- previous_interface: 回到上一界面重新发射，补充当前 lambda_i 的候选池，然后重试当前失败层。若 md.resample_velocities 为 true，补候选时会在上一界面 state 上重新赋速；若为 false，driver 会跳过这类恢复，因为确定性动力学会重复同一条轨迹。
- initial_state: 从初态 A 重新补样本。若 run.initial_models 给了多个独立初态结构，driver 会轮流使用；否则只能从 run.base_model 开始。严格全确定性动力学下，应准备多个独立初态相空间点，而不是重复同一个 base_model。

恢复产生的新候选会计入上游界面的 trials/successes，因此 interfaces.csv 和 summary.json 的概率仍对应实际用过的发射次数。每次恢复都会在 events.jsonl 中写入 event=recovery。

恢复不是替代采样收敛的魔法。若多轮恢复后仍失败，通常说明候选池不足、界面太粗、步数太短，或 CV 不适合描述该区间。

## 15. 温度、压力和系综

FFS 本身不规定温度和系综。它只要求每段短轨迹是无偏动力学，并使用相同的物理条件。若 md.temperature_K 存在，driver 会在生成 run.in 时覆盖 velocity 和常见 ensemble 温度字段；若不存在，则完全使用模板温度。md.resample_velocities 只改变发射初速度，不应引入偏置力。

对于固固相变：

- 提高温度可以降低有效能垒，常用于先验证流程能否通路。
- 压力、晶胞自由度和 barostat 设置会强烈影响相变路径。
- 模板 run.in 中的 ensemble、velocity、pressure 应与目标物理条件一致。
- 生产速率应使用目标温度和压力重新采样，不能直接把高温验证结果当作低温速率。

## 16. 故障诊断

常见问题：

1. 没有 GASCVlog.txt

   FFSampling 可能被写在 run 后面，或者 monitor 没有在 MD 前加载。当前 driver 会把 FFSampling 插入 run 前。

2. flux 阶段没有 crossing

   lambda_0 太高，flux_max_steps 太短，温度太低，或 CV 方向写反。

3. interface trial 全失败

   当前界面间距太大，interface_max_steps 太短，fail_interface 太严格，或该区间能垒过高。

4. 成功但没有 restart.xyz

   gpu-sampling 没有在停止时 dump restart，或进程异常退出。检查 gpu_sampling.out。

5. CV 值异常

   检查 GAScfg.yaml 的 n_atoms、neighbor_rc、max_neighbors 是否与体系和模型一致。

6. 速率时间不准

   driver 优先从 gpu_sampling.out 解析 N steps completed。解析失败时用 GASCVlog 行数乘 cv_log_interval fallback。若需要高精度通量，建议 cv_log_interval 为 1。

7. replica 轨迹完全相同

   检查起点 xyz 是否带 vel 属性、md.resample_velocities 是否为 true、run.in 是否生成 velocity T seed N。确定性 MD 下，同一相点和同一速度会得到完全相同的轨迹。

8. adaptive 到最小宽度后仍失败

   不要继续无限细化。先从上游界面补充更多候选 states；如果上游样本也不足，再逐层往前补，必要时回到 flux 阶段重新收集 crossing states。

## 17. 最小测试流程

先做 smoke test：

    cd /home/tensor/software/GPUMD
    /home/tensor/anaconda3/envs/py38/bin/python FFS_src/ffs_driver.py FFS_src/ffs-test/ffs_smoke.yaml

检查：

- summary.json 存在。
- interfaces.csv 有概率。
- states/interface_000 有 state_*.xyz。
- success_segments 下有成功片段和 restart.xyz。
- trials 中生成的 run.in 里 FFSampling 位于 run 前面。

## 18. 生产运行建议

1. 先用少量界面和短步数做 smoke test。
2. 检查 CV 是否符合预期方向。
3. 设置合理 lambda_0，收集足够 flux crossings。
4. 逐层增加界面，观察每层 probability。
5. 对概率太低的区间插入更多界面。
6. 提高 min_trials 和 min_successes 做正式统计。
7. 保存 summary.json、interfaces.csv、events.jsonl 和 success_segments 作为结果证据。
8. 若使用 adaptive.enabled，检查 summary.json 中最终界面数组和 adaptive.refinements，而不是只看初始 yaml。
9. 若使用 recovery.mode，检查 events.jsonl 中的 recovery 事件，确认补样本来自 previous_interface 还是 initial_state。

## 19. 示例：GaN B4-B1

GaN B4-B1 相变可以把配位数作为 CV。B4 约为 4，B1 约为 6，因此 direction 为 1。

测试配置可以使用较高温度和细界面，例如：

    interfaces: [4.05, 4.20, 4.35, 4.50, 4.65, 4.80, 5.00, 5.20, 5.40, 5.60, 5.80, 5.95]
    fail_interface: 3.95
    direction: 1
    temperature_K: 1000

这只是流程验证示例。真实生产参数需要根据每层通过率继续调界面、步数和发射数量。若某一段完全过不去，可以打开 adaptive.enabled，让 driver 在大于最小宽度的失败区间自动插入中点界面。

本仓库中的 `ffs-test/ffs_1500K_fine_probe.yaml` 是更激进的高温细界面探针。当前测试中它能从 4.10 推进到 4.425，但在 4.425 -> 4.45 区间多次回落到约 4.38-4.39，因此它只证明流程和恢复逻辑可用，不应视作完整 B4-B1 相变结果。


## 20. 算法参考

本手册采用标准 direct FFS 思路：先估计从 A 穿过第一界面的通量，再逐层估计条件通过概率，最后相乘得到速率。算法背景可参考：

- Allen, Valeriani, ten Wolde, Forward flux sampling for rare event simulations, Journal of Physics: Condensed Matter 21, 463102 (2009), arXiv: https://arxiv.org/abs/0906.4758
- Borrero and Escobedo, Optimizing the sampling and staging for simulations of rare events via forward flux sampling schemes, Journal of Chemical Physics 129, 024115 (2008), DOI: https://doi.org/10.1063/1.2943285

这些参考给出的核心实践也适用于本实现：界面应按通过率调节；概率过低的区间需要加密界面或增加 trial；通量统计和条件概率统计要分开收敛。
