# dolanga1_mujoco_sim2sim


## 目录结构

- `deploy_mujoco/sim2sim_core.py`: 核心配置、obs 拼接、`policy_full.pt` 推理、PD 控制
- `deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py`: 可视化运行入口
- `deploy_mujoco/configs/dolanga1.yaml`: 参数模板

## 运行示例

在本目录执行：

# 跑 BarrierDual + DreamWaQ-style actor
```bash
cd /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py \
  --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/Dolanga1.xml \
  --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/2026-05-23_10-43-39_barrier_dreamwaq/exported/policy_full.pt \
  --cmd_x 1.0 \
  --cmd_y 0.0 \
  --cmd_yaw 0.0
  
```
