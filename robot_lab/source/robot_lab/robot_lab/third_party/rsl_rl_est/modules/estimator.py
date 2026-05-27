import torch
import torch.nn as nn
from typing import Type
from tensordict import TensorDict
from robot_lab.third_party.rsl_rl_est.modules.encoders import *
from robot_lab.third_party.rsl_rl_est.modules.decoders import *


class Estimator(nn.Module):
    is_recurrent = False

    def __init__(self,
                 obs: TensorDict,
                 memory_buffer_cfgs,
                 encoder_cfgs,
                 decoder_cfgs,
                 ):
        super(Estimator, self).__init__()
        # TODO: device should get in a better way
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = obs.batch_size

        # create memory buffers
        self.memory_buffers = TensorDict()
        if memory_buffer_cfgs is not None:
            for buf_name, buf_cfg in memory_buffer_cfgs.items():
                self.memory_buffers[buf_name] = torch.zeros(batch_size[0], buf_cfg['dim'], device=device)
            obs.update(self.memory_buffers)
            print(f"Initialized memory buffers: {self.memory_buffers}\n"
                  f"--------------------------------------------------------------------")

        # create encoders
        self.encoders = nn.ModuleDict()
        for enc_name, enc_cfg in encoder_cfgs.items():
            enc_type = enc_cfg.pop("type")
            if enc_cfg['network_cfg'] is not None and 'share_network' in enc_cfg['network_cfg']:
                enc_cfg['network_cfg']['shared_network'] = self.encoders[
                    enc_cfg['network_cfg']['shared_network']].encoder
            enc_class = self.resolve_encoder(enc_type)
            self.encoders[enc_name] = enc_class(obs, **enc_cfg).to(device)
            if getattr(self.encoders[enc_name], "is_recurrent", False):
                self.is_recurrent = True
            obs.update(self.encoders[enc_name].encode(obs))
            hidden_state = self.encoders[enc_name].get_hidden_state()
            obs.update(self.encoders[enc_name].encode(obs, hidden_states=hidden_state))
            if hasattr(self.encoders[enc_name], "_detach_flags"):
                detached = [k for k, v in self.encoders[enc_name]._detach_flags.items() if v]
                non_detached = [k for k, v in self.encoders[enc_name]._detach_flags.items() if not v]
            else:
                detached = []
                non_detached = self.encoders[enc_name].obs_groups

            print(
                f"Initialized encoder '{enc_name}' of type '{enc_type}' with \n"
                f"input obs groups {self.encoders[enc_name].obs_groups}\n"
                f"  ├─ detached obs groups : {detached if detached else 'None'}\n"
                f"  └─ non-detached obs    : {non_detached if non_detached else 'None'}\n"
                f"producing latent '{self.encoders[enc_name].latent_name}' dim= {self.encoders[enc_name].latent_dim}:\n"
                f"{self.encoders[enc_name]}\n"
                f"--------------------------------------------------------------------"
            )

        # create decoders
        self.decoders = nn.ModuleDict()
        for dec_name, dec_cfg in decoder_cfgs.items():
            dec_type = dec_cfg.pop("type")
            dec_class = self.resolve_decoder(dec_type)
            self.decoders[dec_name] = dec_class(obs, **dec_cfg).to(device)
            # TODO: decoder should print more logs
            print(
                f"Initialized decoder '{dec_name}' of type '{dec_type}' with loss weight {self.decoders[dec_name].loss_weight}:\n"
                f"{self.decoders[dec_name]}\n"
                f"--------------------------------------------------------------------")

        self.reset(torch.ones(batch_size[0], dtype=torch.bool, device=device))

    def resolve_encoder(self, encoder_name: str) -> Type[Encoder]:
        encoder_dict: dict[str, Type[Encoder]] = {
            'processor': Processor,
            'memory': MemoryEncoder,
            'cnn_processor': CNNProcessor,
            'self_attention_processor': SelfAttentionProcessor,
            "cross_attention_processor": CrossAttentionProcessor,
            'transformer_processor': TransformerProcessor,
            'unet_processor': UNetProcessor,
            'vae': VAE,
            'kivi_kinesthetic': KiviKinestheticEncoder,
            'kivi_visuospatial': KiviVisuospatialEncoder,
        }
        if encoder_name == 'memory':
            self.is_recurrent = True
        if encoder_name not in encoder_dict:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        return encoder_dict[encoder_name]

    def resolve_decoder(self, decoder_name: str) -> Type[Decoder]:
        decoder_dict: dict[str, Type[Decoder]] = {
            'explicit': ExplicitDecoder,
            'himloco': HimlocoDecoder,
            'implicit': ImplicitDecoder,
            'dreamer': DreamerDecoder,
            'kl': KLDivergenceDecoder,
        }
        if decoder_name not in decoder_dict:
            raise ValueError(f"Unknown decoder: {decoder_name}")
        return decoder_dict[decoder_name]

    def encode(self, obs: TensorDict, dones=None, hidden_states=None) -> TensorDict:
        self.memory_update_list = []
        for memory_name in self.memory_buffers.keys():
            if memory_name not in obs:
                obs[memory_name] = self.memory_buffers[memory_name].detach()
                self.memory_update_list.append(memory_name)
        for enc_name, encoder in self.encoders.items():
            obs.update(encoder.encode(obs, dones=dones, hidden_states=hidden_states))
        for memory_name in self.memory_update_list:
            if memory_name + '_update' in obs:
                self.memory_buffers[memory_name] = obs[memory_name + '_update'].detach()
        return obs

    def encode_inference(self, obs: TensorDict, dones=None, hidden_states=None) -> TensorDict:
        self.memory_update_list = []
        for memory_name in self.memory_buffers.keys():
            if memory_name not in obs:
                obs[memory_name] = self.memory_buffers[memory_name].detach()
                self.memory_update_list.append(memory_name)
        for enc_name, encoder in self.encoders.items():
            obs.update(encoder.encode_inference(obs, dones=dones, hidden_states=hidden_states))
        for memory_name in self.memory_update_list:
            if memory_name + '_update' in obs:
                self.memory_buffers[memory_name] = obs[memory_name + '_update'].detach()
        return obs

    def loss_fn(self, obs: TensorDict) -> TensorDict:
        loss_dict = TensorDict({}, batch_size=obs.batch_size)
        for dec_name, decoder in self.decoders.items():
            loss_dict.update({dec_name: decoder.loss_fn(obs)})
        loss_dict.update({'total_loss': sum(loss_dict.values())})
        return loss_dict

    def reset(self, dones: torch.Tensor, **kwargs):
        for encoder_name in self.encoders:
            self.encoders[encoder_name].reset(dones)

        for memory_name in self.memory_buffers.keys():
            self.memory_buffers[memory_name][dones] = 0.0

    def get_hidden_states(self):
        hidden_states = {}
        for encoder_name in self.encoders:
            hidden_states.update(self.encoders[encoder_name].get_hidden_state())
        return hidden_states

    def get_hidden_states_inference(self):
        hidden_states = {}
        for encoder_name in self.encoders:
            hidden_states.update(self.encoders[encoder_name].get_hidden_state_inference())
        return hidden_states
