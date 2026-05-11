import torch
from .encoder import Encoder
from robot_lab.third_party.rsl_rl_est.networks import Memory



class MemoryEncoder(Encoder):
    def __init__(self,
                 obs,
                 obs_groups: list[str | tuple[str, bool]],
                 latent_name,
                 latent_dim,
                 network_cfg,
                 **kwargs):
        super(MemoryEncoder, self).__init__(
            obs=obs,
            obs_groups=obs_groups,
            latent_name=latent_name,
            latent_dim=latent_dim)
        if kwargs and any(value is not None for value in kwargs.values()):
            print(
                "MemoryEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys() if kwargs[key] is not None])
            )

        self.encoder = Memory(
            input_size=self.num_obs,
            type=network_cfg['type'],
            num_layers=network_cfg['num_layers'],
            hidden_size=self.latent_dim,
        )
        self.num_transitions_per_env: int = network_cfg['num_transitions_per_env']

    def encode(self, obs, dones=None, hidden_states=None):
        encoder_obs = self.get_obs_tensor(obs)
        if dones is not None:
            encoder_obs = encoder_obs.view(self.num_transitions_per_env, -1, self.num_obs)
            dones = dones.view(self.num_transitions_per_env, -1, 1)
        if hidden_states is not None:
            hidden_states = hidden_states.get(self.latent_name, None)
        out = self.encoder(encoder_obs, dones, hidden_states).squeeze(0)
        return {self.latent_name: out}

    def encode_inference(self, obs, dones=None, hidden_states=None):
        return self.encode(obs, dones, hidden_states)

    def get_hidden_state(self):
        return {self.latent_name: self.encoder.hidden_states}

    def get_hidden_state_inference(self):
        return {self.latent_name: self.encoder.hidden_states_inference}

    def reset(self, done=None):
        self.encoder.reset(done)