import torch
import torch.nn as nn
from .decoder import Decoder
try:
    from rsl_rl.networks import MLP
except ImportError:
    from rsl_rl.modules import MLP
import torch.nn.functional as F


class HimlocoDecoder(Decoder):
    def __init__(self, obs,
                 obs_groups,
                 target_groups,
                 network_cfg,
                 loss_weight=1.0,
                 loss_type='mse',
                 add_on_cfg=None,
                 **kwargs):
        super(HimlocoDecoder, self).__init__(
            obs=obs,
            obs_groups=obs_groups,
            target_groups=target_groups,
            loss_weight=loss_weight,
        )
        if kwargs and any(value is not None for value in kwargs.values()):
            print(
                "HimlocoDecoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys() if kwargs[key] is not None])
            )

        num_prototype = add_on_cfg.get('num_prototype', None)
        temperature = add_on_cfg.get('temperature', None)
        if num_prototype is None or temperature is None:
            raise Exception("HimlocoDecoder.__init__ add_on_cfg is None or missing keys")
        self.proto = nn.Embedding(num_prototype, self.num_obs)
        self.temperature = temperature

        if network_cfg is None:
            raise Exception("HimlocoDecoder.__init__ network_cfg is None")
        self.decoder = MLP(self.num_targets, self.num_obs, network_cfg['hidden_dims'], activation=network_cfg['activation'],)

    def loss_fn(self, obs, **kwargs):
        decoder_obs = self.get_decoder_obs_tensor(obs)
        decoder_targets = self.get_decoder_targets_tensor(obs)

        z_t = self.decoder(decoder_targets)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = decoder_obs @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean(-1)

        return loss


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T
