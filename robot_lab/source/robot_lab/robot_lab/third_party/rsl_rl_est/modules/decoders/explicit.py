import torch.nn as nn
from .decoder import Decoder

class ExplicitDecoder(Decoder):
    def __init__(self, obs,
                 obs_groups,
                 target_groups,
                 loss_weight=1.0,
                 loss_type='mse',
                 **kwargs):
        super(ExplicitDecoder, self).__init__(
            obs=obs,
            obs_groups=obs_groups,
            target_groups=target_groups,
            loss_weight=loss_weight,
        )
        if kwargs and any(value is not None for value in kwargs.values()):
            print(
                "ExplicitDecoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys() if kwargs[key] is not None])
            )
        if loss_type == 'mse':
            self.f = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.f = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def loss_fn(self, obs, **kwargs):
        decoder_obs = self.get_decoder_obs_tensor(obs)
        decoder_targets = self.get_decoder_targets_tensor(obs)
        loss = self.f(decoder_obs, decoder_targets).mean(-1)
        return loss * self.loss_weight