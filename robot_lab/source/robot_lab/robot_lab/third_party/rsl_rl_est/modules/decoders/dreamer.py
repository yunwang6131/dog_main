import torch
import torch.distributions as torchd
from .decoder import Decoder


class DreamerDecoder(Decoder):
    """
    Dreamer-style symmetric KL decoder with add_on_cfg support.

    posterior 来自 obs_groups[0]
    prior     来自 target_groups[0]

    需要 obs 中存在:
      <name>_mu
      <name>_logvar

    支持:
      beta_kl
      free_nats
      logvar_min / logvar_max
      alpha（对称 KL 权重）
    """

    def __init__(
        self,
        obs,
        obs_groups,
        target_groups,
        loss_weight: float = 1.0,
        loss_type: str = "dreamer",
        add_on_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            target_groups=target_groups,
            loss_weight=loss_weight,
        )

        # -----------------------------
        #  默认参数
        # -----------------------------
        cfg_defaults = {
            "beta_kl": 1.0,
            "free_nats": 1.0,
            "logvar_min": -10.0,
            "logvar_max": 4.0,
            "alpha": 0.5,
        }

        # -----------------------------
        #  从 add_on_cfg 覆盖
        # -----------------------------
        add_on_cfg = add_on_cfg or {}

        # 检查 add_on_cfg 中未知字段
        unknown = [k for k in add_on_cfg.keys() if k not in cfg_defaults]
        if unknown:
            print(
                f"DreamerDecoder Warning: Unknown add_on_cfg fields ignored: {unknown}"
            )

        # 正式赋值
        for k, v in cfg_defaults.items():
            setattr(self, k, float(add_on_cfg.get(k, v)))

        # -----------------------------
        #  loss_type 校验
        # -----------------------------
        if loss_type != "dreamer":
            raise ValueError(f"DreamerDecoder only supports loss_type='dreamer', got {loss_type}")

        assert len(self.obs_groups) >= 1, "DreamerDecoder requires at least one posterior obs_group"
        assert len(self.target_groups) >= 1, "DreamerDecoder requires at least one prior target_group"

        # -----------------------------
        #  stop-grad helper
        # -----------------------------
        def _sg(d):
            if isinstance(d, torchd.Independent):
                return torchd.Independent(_sg(d.base_dist), d.reinterpreted_batch_ndims)
            if isinstance(d, torchd.Normal):
                return torchd.Normal(d.mean.detach(), d.stddev.detach())
            if hasattr(d, "mean") and hasattr(d, "stddev"):
                return type(d)(d.mean.detach(), d.stddev.detach())
            if hasattr(d, "logits"):
                return type(d)(logits=d.logits.detach())
            if hasattr(d, "probs"):
                return type(d)(probs=d.probs.detach())
            raise TypeError(f"Unsupported distribution type for stop-grad: {type(d)}")

        # 对称 KL
        self._kl = lambda post, prior: (
            self.alpha * torchd.kl.kl_divergence(_sg(post), prior)
            # + (1 - self.alpha) * torchd.kl.kl_divergence(post, _sg(prior))
        )

    # -------------------------------------------------------
    #  helpers
    # -------------------------------------------------------
    @staticmethod
    def _name(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def _get_post_prior(self, obs):
        post_name = self._name(self.obs_groups[0])
        prior_name = self._name(self.target_groups[0])

        mu_post = obs[f"{post_name}_mu"]
        logvar_post = obs[f"{post_name}_logvar"]
        mu_prior = obs[f"{prior_name}_mu"]
        logvar_prior = obs[f"{prior_name}_logvar"]

        # clamp
        logvar_post = torch.clamp(logvar_post, self.logvar_min, self.logvar_max)
        logvar_prior = torch.clamp(logvar_prior, self.logvar_min, self.logvar_max)

        std_post = torch.exp(0.5 * logvar_post)
        std_prior = torch.exp(0.5 * logvar_prior)

        post = torchd.Independent(torchd.Normal(mu_post, std_post), 1)
        prior = torchd.Independent(torchd.Normal(mu_prior, std_prior), 1)
        return post, prior

    # -------------------------------------------------------
    #  public API
    # -------------------------------------------------------
    def loss_fn(self, obs, **kwargs):
        post, prior = self._get_post_prior(obs)

        kl = self._kl(post, prior)  # shape [B]

        # free-nats
        if self.free_nats > 0:
            kl = torch.clamp_min(kl - self.free_nats, 0.0)

        loss = self.beta_kl * kl
        return loss