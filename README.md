## 运行指令
cd /home/sen/wy/barrier_main/robot_lab

python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0 \
  --headless \
  --device cuda:0 \
  --logger wandb \
  --log_project_name barrier \
  --run_name kp80_kd20


python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0