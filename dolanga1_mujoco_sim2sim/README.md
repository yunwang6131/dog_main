# dolanga1_mujoco_sim2sim

独立于 `robot_lab` 的 MuJoCo sim2sim 骨架工程。

## 目录结构

- `deploy_mujoco/sim2sim_core.py`: 核心配置、obs 拼接、ONNX 推理、PD 控制
- `deploy_mujoco/sim2sim_dolanga1_trot.py`: 无界面运行入口
- `deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py`: 可视化运行入口
- `deploy_mujoco/configs/dolanga1.yaml`: 参数模板

## 运行示例

在本目录执行：

```bash
python deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py \
  --load_model /path/to/scene.xml \
  --policy /path/to/policy.onnx
```

## 注意

当前 `sim2sim_core.py` 仍有一处 `TODO`:

- 关节 name 到 MuJoCo `qpos/qvel` 索引的稳健映射（现在是占位顺序映射）。

建议下一步先完成该映射，再开始系统调参。
