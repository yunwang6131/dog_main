# dolanga1_mujoco_sim2sim


## 目录结构

- `deploy_mujoco/sim2sim_core.py`: 核心配置、obs 拼接、`policy_full.pt` 推理、PD 控制
- `deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py`: 可视化运行入口
- `deploy_mujoco/configs/dolanga1.yaml`: 参数模板

## 运行示例

在本目录执行：

# 跑 BarrierDual + DreamWaQ-style actor

sim2sim 入口使用 merged TorchScript `policy_full.pt`，输入为 `history, current`。不要直接传 `policy.onnx`。

# 施加外力的自建窗口
```bash
cd /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/<run>/exported/policy_full.pt   --cmd_x 0.5   --cmd_y 0.0   --cmd_yaw 0.0
  
```
# 原始窗口
```bash

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/<run>/exported/policy_full.pt   --cmd_x 0.5   --cmd_y 0.0   --cmd_yaw 0.0
  
```
