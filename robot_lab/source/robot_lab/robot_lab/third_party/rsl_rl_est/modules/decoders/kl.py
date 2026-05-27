import torch

from .decoder import Decoder


class KLDivergenceDecoder(Decoder):
    """KL(q(z|x) || N(0, I)) loss for VAE-style encoder outputs."""

    def __init__(
        self,
        obs,
        obs_groups,
        target_groups=None,
        loss_weight=1.0,
        loss_type=None,
        network_cfg=None,
        add_on_cfg=None,
        **kwargs,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            target_groups=target_groups or [],
            loss_weight=loss_weight,
        )
        if len(self.obs_groups) != 2:
            raise ValueError("KLDivergenceDecoder expects obs_groups=[mu, logvar].")

    def loss_fn(self, obs, **kwargs):
        mu, logvar = self.get_decoder_obs_list(obs)
        loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)
        return loss * self.loss_weight
