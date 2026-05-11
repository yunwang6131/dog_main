import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from tensordict import TensorDict


class PPOWM(PPO):
    policy: ActorCritic | ActorCriticRecurrent
    """The actor critic module."""

    def __init__(
            self,
            policy: ActorCritic | ActorCriticRecurrent,
            num_learning_epochs: int = 5,
            num_mini_batches: int = 4,
            clip_param: float = 0.2,
            gamma: float = 0.99,
            lam: float = 0.95,
            value_loss_coef: float = 1.0,
            entropy_coef: float = 0.01,
            learning_rate: float = 0.001,
            max_grad_norm: float = 1.0,
            use_clipped_value_loss: bool = True,
            schedule: str = "adaptive",
            desired_kl: float = 0.01,
            device: str = "cpu",
            normalize_advantage_per_mini_batch: bool = False,
            # RND parameters
            rnd_cfg: dict | None = None,
            # Symmetry parameters
            symmetry_cfg: dict | None = None,
            # Distributed training parameters
            multi_gpu_cfg: dict | None = None,
    ) -> None:
        super(PPOWM, self).__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()