from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
import torch
from collections.abc import Generator


class RolloutStorageEst(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.est_hidden_states = None  # RNN 的 hidden states
            self.est_hidden_states_mirror = None  # RNN 的 hidden states（镜像版）

    def __init__(
            self,
            training_type: str,
            num_envs: int,
            num_transitions_per_env: int,
            obs: TensorDict,
            actions_shape: tuple[int] | list[int],
            device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.saved_est_hidden_states = {}
        self.saved_est_hidden_states_mirror = {}

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if not hasattr(self, "mu"):
                self.mu = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *transition.action_mean.shape[1:], device=self.device
                )
                self.sigma = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *transition.action_sigma.shape[1:], device=self.device
                )
            if self.distribution_params is None:
                self.distribution_params = (
                    torch.zeros(
                        self.num_transitions_per_env,
                        self.num_envs,
                        *transition.action_mean.shape[1:],
                        device=self.device,
                    ),
                    torch.zeros(
                        self.num_transitions_per_env,
                        self.num_envs,
                        *transition.action_sigma.shape[1:],
                        device=self.device,
                    ),
                )
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
            self.distribution_params[0][self.step].copy_(transition.action_mean)
            self.distribution_params[1][self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)
        self._save_est_hidden_states(transition.est_hidden_states)
        self._save_est_hidden_states_mirror(transition.est_hidden_states_mirror)

        # Increment the counter
        self.step += 1

    def compute_returns(
            self,
            last_values: torch.Tensor,
            gamma: float,
            lam: float,
            normalize_advantage: bool = True,
    ) -> None:
        """Compute GAE returns for rsl-rl versions that no longer expose this helper."""
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        for batch in super().mini_batch_generator(num_mini_batches, num_epochs):
            old_mu_batch, old_sigma_batch = batch.old_distribution_params
            yield (
                batch.observations,
                batch.actions,
                batch.values,
                batch.advantages,
                batch.returns,
                batch.old_actions_log_prob,
                old_mu_batch,
                old_sigma_batch,
                (None, None),
                None,
            )

    def _save_est_hidden_states(self, est_hidden_states):
        if est_hidden_states is None:
            return

        for key, value in est_hidden_states.items():
            hid = value if isinstance(value, tuple) else (value,)
            # initialize if needed
            if key not in self.saved_est_hidden_states:
                self.saved_est_hidden_states[key] = [
                    torch.zeros(self.observations.shape[0], *hid[i].shape, device=self.device) for i in
                    range(len(hid))
                ]
            # copy the states
            for i in range(len(hid)):
                self.saved_est_hidden_states[key][i][self.step].copy_(hid[i])

    def _save_est_hidden_states_mirror(self, est_hidden_states_mirror):
        if est_hidden_states_mirror is None:
            return

        for key, value in est_hidden_states_mirror.items():
            hid = value if isinstance(value, tuple) else (value,)
            # initialize if needed
            if key not in self.saved_est_hidden_states_mirror:
                self.saved_est_hidden_states_mirror[key] = [
                    torch.zeros(self.observations.shape[0], *hid[i].shape, device=self.device) for i in
                    range(len(hid))
                ]
            # copy the states
            for i in range(len(hid)):
                self.saved_est_hidden_states_mirror[key][i][self.step].copy_(hid[i])

    def recurrent_est_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = self.dones[:, start:stop].flatten(0, 1)
                obs_batch = self.observations[:, start:stop].flatten(0, 1)
                actions_batch = self.actions[:, start:stop].flatten(0, 1)
                old_mu_batch = self.mu[:, start:stop].flatten(0, 1)
                old_sigma_batch = self.sigma[:, start:stop].flatten(0, 1)
                returns_batch = self.returns[:, start:stop].flatten(0, 1)
                advantages_batch = self.advantages[:, start:stop].flatten(0, 1)
                values_batch = self.values[:, start:stop].flatten(0, 1)
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop].flatten(0, 1)

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_batch = {}

                for key, saved_hidden_states in self.saved_est_hidden_states.items():
                    hid_e_batch = [
                        saved_hidden_states[i].permute(2, 0, 1, 3)[last_was_done][
                            first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for i in range(len(saved_hidden_states))
                    ]
                    # remove the tuple for GRU
                    hid_e_batch = hid_e_batch[0] if len(hid_e_batch) == 1 else tuple(hid_e_batch)
                    hid_batch[key] = hid_e_batch

                hid_batch_mirror = {}
                for key, saved_hidden_states in self.saved_est_hidden_states_mirror.items():
                    hid_e_batch_mirror = [
                        saved_hidden_states[i].permute(2, 0, 1, 3)[last_was_done][
                            first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for i in range(len(saved_hidden_states))
                    ]
                    # remove the tuple for GRU
                    hid_e_batch_mirror = hid_e_batch_mirror[0] if len(hid_e_batch_mirror) == 1 else tuple(
                        hid_e_batch_mirror)
                    hid_batch_mirror[key] = hid_e_batch_mirror

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_batch,
                    hid_batch_mirror,
                ), masks_batch

                first_traj = last_traj
