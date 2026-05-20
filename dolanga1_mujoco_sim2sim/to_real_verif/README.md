# to_real_verif

这个目录单独用来验证 real-robot handover bundle，先固定校验：

- `policy_full.pt`
- `sim2real_config.yaml`

另外也放一套单独的 MuJoCo 可视化入口，默认走：

- `resources/a1_scene.xml`
- `policy_full.pt`
- `init_base_height = 0.47`
- `q_default calf = -1.6`

当前默认目标是：

- `/home/sen/wy/dog_main/robot_lab/logs/rsl_rl/dolanga1_rough_dreamwaq/2026-05-13_17-33-33_initial/exported`

## 目录内容

- `verify_bundle.py`: 通用校验脚本。
- `run_2026-05-13_17-33-33_initial.sh`: 针对当前 bundle 的一键入口。
- `dreamwaq_dog_viewer_2026-05-13_17-33-33_initial.yaml`: 本地 MuJoCo viewer 配置。
- `run_dreamwaq_dog_viewer.py`: MuJoCo viewer / smoke 入口。
- `run_2026-05-13_17-33-33_initial_dog_viewer.sh`: 用 `dog` 环境启动 viewer。
- `reports/`: 生成的验证报告。

## 默认做的检查

- `policy_full.pt` 能否直接加载并前向。
- `policy_full.pt` 是否和 `cenet.pt + policy.pt` 数值一致。
- `sim2real_config.yaml` 自身维度和控制合同是否自洽。
- `sim2real_config.yaml` 和同次训练的 `params/env.yaml` 是否对齐。

## 运行

```bash
cd /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim
bash to_real_verif/run_2026-05-13_17-33-33_initial.sh
```

如果你想换 Python：

```bash
cd /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim
PYTHON_BIN=/path/to/python bash to_real_verif/run_2026-05-13_17-33-33_initial.sh
```

报告默认会写到：

- `to_real_verif/reports/2026-05-13_17-33-33_initial_report.md`

## MuJoCo 可视化

默认用 `dog` 环境，因为这套环境同时有 `mujoco + onnxruntime + torch`。
现在这条链路直接加载 `policy_full.pt`，不再依赖拆开的 `policy.onnx + cenet.pt`。

开 MuJoCo viewer：

```bash
cd /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim
bash to_real_verif/run_2026-05-13_17-33-33_initial_dog_viewer.sh
```

纯键盘控制：

```bash
cd /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim
bash to_real_verif/run_2026-05-13_17-33-33_initial_dog_viewer.sh --keyboard
```

如果你只想先做无界面 smoke：

```bash
cd /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim
bash to_real_verif/run_2026-05-13_17-33-33_initial_dog_viewer.sh --no_viewer --smoke_steps 8
```

如果你想临时改单腿默认角或初始高度：

```bash
bash to_real_verif/run_2026-05-13_17-33-33_initial_dog_viewer.sh --init_base_height 0.47 --calf_default -1.6
```

这条 viewer 链路实际直接使用 [policy_full.pt](/home/sen/wy/dog_main/robot_lab/logs/rsl_rl/dolanga1_rough_dreamwaq/2026-05-13_17-33-33_initial/exported/policy_full.pt:1)。
