## 运行指令
# 跑PPO

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-Dolanga1-v0 --headless --logger wandb --log_project_name dolang --run_name dog_test
```
# 跑dreamwaq

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0 --headless --logger wandb --log_project_name dog_Dwaq --run_name initial
```
```bash
/home/dl/sim/IsaacLab/_isaac_sim/kit/python/bin/python3 scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0
```

```bash
/home/dl/sim/IsaacLab/_isaac_sim/kit/python/bin/python3 -m pip install -e .
```