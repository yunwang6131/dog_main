## 运行指令

# 训练
cd /home/sen/wy/barrier_main/robot_lab

python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0 \
  --headless \
  --device cuda:0 \
  --logger wandb \
  --log_project_name dreamwaq \
  --run_name inital

# Play

python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0

# Sim2sim自建窗口（可以手动施加外力）
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_dreamwaq/<run>/exported/policy_full.pt   --cmd_x 1.0 --cmd_y 0.0 --cmd_yaw 0.0

# sim2sim 系统窗口

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_dreamwaq/<run>/exported/policy_full.pt   --cmd_x 0.5   --cmd_y 0.0   --cmd_yaw 0.0

# 直接用isaaclab自带的程序训练

cd /home/dl/sim/IsaacLab
./isaaclab.sh -p /home/dl/wy/barrier_main/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0 \
  --headless \
  --logger wandb \
  --log_project_name dreamwaq \
  --run_name initial

# play 

./isaaclab.sh -p /home/dl/wy/barrier_main/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py   --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0 

# sim2sim 要log的真实地址
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py \
  --load_model /home/dl/wy/barrier_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml \
  --policy /home/dl/sim/IsaacLab/logs/rsl_rl/dolanga1_rough_dreamwaq/2026-05-28_17-31-59_change_height/exported/policy_full.pt \
  --cmd_x 0.5 \
  --cmd_y 0.0 \
  --cmd_yaw 0.0

cd /home/dl/wy/barrier_main/dolanga1_mujoco_sim2sim

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py \
  --load_model /home/dl/wy/barrier_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml \
  --policy /home/dl/sim/IsaacLab/logs/rsl_rl/dolanga1_rough_dreamwaq/2026-05-28_17-31-59_change_height/exported/policy_full.pt \
  --cmd_x 1.0 \
  --cmd_y 0.0 \
  --cmd_yaw 0.0

./isaaclab.sh -p /home/dl/wy/barrier_main/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-DreamWaQ-v0 --headless --device cuda:0 --logger wandb --log_project_name dreamwaq --resume --load_run 2026-06-04_11-55-59 --checkpoint model_1700.pt

# 也可以用tensorboard看

tensorboard --logdir /home/dl/sim/IsaacLab/logs/rsl_rl/dolanga1_rough_dreamwaq/2026-06-10_09-42-43