from .decoder import Decoder
try:
    from rsl_rl.networks import MLP
except ImportError:
    from rsl_rl.modules import MLP
import torch.nn as nn


class ImplicitDecoder(Decoder):
    def __init__(self, obs,
                 obs_groups,
                 target_groups,
                 network_cfg,
                 loss_weight=1.0,
                 loss_type='mse',
                 add_on_cfg=None,
                 **kwargs):
        super(ImplicitDecoder, self).__init__(
            obs=obs,
            obs_groups=obs_groups,
            target_groups=target_groups,
            loss_weight=loss_weight,
        )
        if kwargs and any(value is not None for value in kwargs.values()):
            print(
                "ImplicitDecoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys() if kwargs[key] is not None])
            )
        if loss_type == 'mse':
            self.f = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.f = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.decoder = MLP(self.num_obs, self.num_targets, network_cfg['hidden_dims'],
                           activation=network_cfg['activation'], )

    def loss_fn(self, obs, **kwargs):
        decoder_obs = self.get_decoder_obs_tensor(obs)
        decoder_targets = self.get_decoder_targets_tensor(obs)
        decoder_outputs = self.decoder(decoder_obs)
        loss = self.f(decoder_outputs, decoder_targets).mean(-1)
        return loss
