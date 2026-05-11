from __future__ import annotations

from isaaclab_rl.rsl_rl import *


@configclass
class MemoryBufferCfg:
    dim: int = MISSING
    """Dimensionality of the memory buffer."""

@configclass
class EncoderCfg:
    type: str = MISSING
    """Type of encoder to use, e.g., 'processor', 'contact_encoder', 'ae'."""

    obs_groups: list[tuple | str] = MISSING
    """List of observation groups to be processed by the encoder."""

    latent_name: str | None = MISSING
    """Name of the latent variable produced by the encoder."""

    latent_dim: int = MISSING
    """Dimensionality of the latent space produced by the encoder."""

    network_cfg: None = None
    """Configuration for the network used in the encoder, such as hidden dimensions and normalization settings."""


@configclass
class DecoderCfg:
    type: str = MISSING
    """Type of decoder to use, e.g., 'explicit', 'vae_decoder'."""

    obs_groups: list[tuple | str] = MISSING
    """List of observation groups to be used as input for the decoder."""

    target_groups: list[tuple | str] = MISSING
    """List of target observation groups that the decoder aims to reconstruct."""

    loss_weight: float = 1.0
    """Weight of the decoder's loss in the overall training objective."""

    loss_type: str = "mse"
    """Type of loss function to use for the decoder, e.g., 'mse', 'bce'."""

    network_cfg: None = None
    """Configuration for the network used in the decoder, such as hidden dimensions and activation functions."""

    add_on_cfg: None = None
    """Additional configuration specific to certain decoder types, e.g., HimlocoDecoder."""


@configclass
class EstimatorCfg:
    memory_buffer_cfgs: dict[str, MemoryBufferCfg] = MISSING
    """Dictionary of memory buffer configurations."""

    encoder_cfgs: dict[str, EncoderCfg] = MISSING
    """Dictionary of encoder configurations."""

    decoder_cfgs: dict[str, DecoderCfg] = MISSING
    """Dictionary of decoder configurations."""


@configclass
class RslRlEstPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with RSL-EST."""

    class_name: str = "ActorCriticEst"
    """The policy class name. Default is ActorCriticEst."""

    estimator_cfg: EstimatorCfg | None = None
    """The estimator configuration."""


@configclass
class RslRlEstPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with RSL-EST."""

    class_name: str = "PPOEst"
    """The algorithm class name. Default is PPOEst."""

    obs_go_next: list[str] | None = None
    """List of observation groups for next observations."""


@configclass
class RslRlEstWMAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with RSL-EST."""

    class_name: str = "Dreamer"
    """The algorithm class name. Default is Dreamer."""

    obs_go_next: list[str] | None = None
    """List of observation groups for next observations."""

    obs_replace_dict: dict[str, str] | None = None
    """Dictionary for replacing observation groups."""

    num_dreamer_steps_per_env : int = 1
    """Number of dreamer steps per environment."""


@configclass
class RslRlEstOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runners for on-policy algorithms."""

    class_name: str = "OnPolicyRunnerEst"
    """The runners class name. Default is OnPolicyRunner."""

    policy: RslRlEstPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlEstPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
