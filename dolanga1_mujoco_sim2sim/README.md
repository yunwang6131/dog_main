# dolanga1_mujoco_sim2sim


## 目录结构

- `deploy_mujoco/sim2sim_core.py`: 核心配置、obs 拼接、ONNX 推理、PD 控制
- `deploy_mujoco/sim2sim_dolanga1_trot.py`: 无界面运行入口
- `deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py`: 可视化运行入口
- `deploy_mujoco/configs/dolanga1.yaml`: 参数模板

# 另外一套是跑dramwaq的，相互独立

## 运行示例

在本目录执行：
# 跑dreamwaq
```bash
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_dreamwaq_viewer.py   --load_model /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml --policy /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/models/2026-05-13_17-33-33_initial/exported/policy.onnx --cenet /home/sen/wy/dog_main/robot_lab/logs/rsl_rl/dolanga1_rough_dreamwaq/2026-05-13_17-33-33_initial/exported/cenet.pt
```

# 跑PPO
```bash
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py   --load_model /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/models/2026-05-13_15-27-00_change_urdf/exported/policy.onnx
```

## sim2sim 扰动 YAML

```bash
PYTHONPATH=. python /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/deploy_mujoco_viewer/sim2sim_dolanga1_dreamwaq_viewer.py   --load_model /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/models/2026-05-13_17-33-33_initial/exported/policy.onnx --cenet /home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/models/2026-05-13_17-33-33_initial/exported/cenet.pt --perturb_yaml deploy_mujoco/configs/sim2sim_perturb.yaml
```

编辑 `deploy_mujoco/configs/sim2sim_perturb.yaml`，启动时加：

```bash
--perturb_yaml deploy_mujoco/configs/sim2sim_perturb.yaml
```

