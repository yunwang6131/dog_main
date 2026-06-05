## 运行指令

# 训练
cd /home/sen/wy/barrier_main/robot_lab

python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0 \
  --headless \
  --device cuda:0 \
  --logger wandb \
  --log_project_name barrier_waq \
  --run_name inital

# Play

python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0

# Sim2sim自建窗口（可以手动施加外力）
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/<run>/exported/policy_full.pt   --cmd_x 1.0 --cmd_y 0.0 --cmd_yaw 0.0

# sim2sim 系统窗口

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/<run>/exported/policy_full.pt   --cmd_x 0.5   --cmd_y 0.0   --cmd_yaw 0.0

# 直接用isaaclab自带的程序训练

cd /home/dl/sim/IsaacLab
./isaaclab.sh -p /home/dl/wy/barrier_main/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0 \
  --headless \
  --logger wandb \
  --log_project_name barrier_waq \
  --run_name initial

# play 

./isaaclab.sh -p /home/dl/wy/barrier_main/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py   --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0 

# sim2sim 要log的真实地址
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py \
  --load_model /home/dl/wy/barrier_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml \
  --policy /home/dl/sim/IsaacLab/logs/rsl_rl/dolanga1_rough_barrier_dual/2026-05-28_17-31-59_change_height/exported/policy_full.pt \
  --cmd_x 0.5 \
  --cmd_y 0.0 \
  --cmd_yaw 0.0

cd /home/dl/wy/barrier_main/dolanga1_mujoco_sim2sim

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py \
  --load_model /home/dl/wy/barrier_main/dolanga1_mujoco_sim2sim/resources/a1_scene.xml \
  --policy /home/dl/sim/IsaacLab/logs/rsl_rl/dolanga1_rough_barrier_dual/2026-05-28_17-31-59_change_height/exported/policy_full.pt \
  --cmd_x 1.0 \
  --cmd_y 0.0 \
  --cmd_yaw 0.0
# humanoid 冒烟测试

HYDRA_FULL_ERROR=1 python3 scripts/reinforcement_learning/rsl_rl/train.py   --task RobotLab-Isaac-Velocity-Flat-DolangH1-v0   --headless   --num_envs 16   --max_iterations 1   --device cuda:0 2>&1 | tee error.log

# humanoid 训练
python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-DolangH1-v0 \
  --headless \
  --device cuda:0 \
  --logger wandb \
  --log_project_name humanoid \
  --max_iteration 20000 \
  --num_envs 1024

# humanoid play

python3 scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-DolangH1-v0 \
  --num_envs 16 \
  --device cuda:0

python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-DolangH1-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --logger wandb \
  --log_project_name humanoid \
  --resume \
  --load_run 2026-06-04_15-56-05 \
  --checkpoint model_9800.pt

# humanoid sim2sim

cd /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim
PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py \
  --config deploy_mujoco/configs/dolangh1.yaml \
  --mouse_force_body base_link

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py \
  --config deploy_mujoco/configs/dolangh1.yaml \
  --cmd_x 0.3 --cmd_y 0.0 --cmd_yaw 0.0