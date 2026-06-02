## 运行指令
cd /home/sen/wy/barrier_main/robot_lab

python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0 \
  --headless \
  --device cuda:0 \
  --logger wandb \
  --log_project_name barrier \
  --run_name kp80_kd2


python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-BarrierDual-v0

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_dolanga1_trot_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/<run>/exported/policy_full.pt   --cmd_x 1.0 --cmd_y 0.0 --cmd_yaw 0.0

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_no_exF_viewer.py   --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml   --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_barrier_dual/<run>/exported/policy_full.pt   --cmd_x 0.5   --cmd_y 0.0   --cmd_yaw 0.0


python3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-KiVi-v0 \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --logger wandb \
  --log_project_name kivi \
  --run_name kivi \
  --num_envs 256 

python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Dolanga1-KiVi-v0

# 不遮挡视觉

python scripts/reinforcement_learning/rsl_rl/play.py   --task RobotLab-Isaac-Velocity-Rough-Dolanga1-KiVi-v0   --num_envs 16 --keyboard

# 遮挡视觉

python scripts/reinforcement_learning/rsl_rl/play.py   --task RobotLab-Isaac-Velocity-Rough-Dolanga1-KiVi-v0   --num_envs 16   --kivi_ablate_depth zero --keyboard

# KiVi 导出 merged policy_full.pt

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Dolanga1-KiVi-v0 \
  --num_envs 16 \
  --enable_cameras

# KiVi sim2sim 加载 policy_full.pt

cd /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim

PYTHONPATH=. python3 deploy_mujoco_viewer/sim2sim_kivi_viewer.py \
  --load_model /home/wangyun/dog_main_test_barrier/dolanga1_mujoco_sim2sim/resources/a1_scene.xml \
  --policy /home/wangyun/dog_main_test_barrier/robot_lab/logs/rsl_rl/dolanga1_rough_kivi/<run>/exported/policy_full.pt \
  --cmd_x 0.5 \
  --cmd_y 0.0 \
  --cmd_yaw 0.0
