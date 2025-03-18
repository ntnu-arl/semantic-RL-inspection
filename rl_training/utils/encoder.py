from typing import Dict, List, Optional

import torch
from gymnasium import spaces
from torch import Tensor, nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, model_device, nonlinearity
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Encoder(ModelModule):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out

class ResBlock3D(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv3d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv3d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        log.debug("Num input channels: %d", input_ch)

        if cfg.encoder_conv_architecture == "resnet_impala":
            # configuration from the IMPALA paper
            resnet_conf = [[2, 2], [4, 2], [4, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_architecture}")

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = nonlinearity(cfg)
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        log.debug(f"Convolutional layer output size: {self.conv_head_out_size}")

        self.mlp_layers = create_mlp(cfg.encoder_conv_mlp_layers, self.conv_head_out_size, activation)

        # should we do torch.jit here?

        self.encoder_out_size = calc_num_elements(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class Resnet3DEncoder(Encoder):
    def __init__(self, cfg, obs_space, type):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        log.debug("Num input channels: %d", input_ch)

        if type == "occupancy":
            # configuration from the IMPALA paper
            resnet_conf = [[8, 2], [16, 2], [16, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_map_architecture}")

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv3d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock3D(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = nonlinearity(cfg)
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        log.debug(f"Convolutional layer output size: {self.conv_head_out_size}")

        # should we do torch.jit here?
        if type == "occupancy":
            self.mlp_layers = create_mlp(cfg.encoder_conv_map_occupancy_mlp_layers, self.conv_head_out_size, activation)
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_map_mlp_architecture}")
        
        self.encoder_out_size = calc_num_elements(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_img_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:

    if cfg.encoder_conv_architecture.startswith("resnet"):
        return ResnetEncoder(cfg, obs_space)
    else:
        raise NotImplementedError(f"Unknown convolutional architecture {cfg.encoder_conv_architecture}")

def make_map_encoder(cfg: Config, obs_space: ObsSpace, type) -> Encoder:
    """Make (most likely convolutional) encoder for 3Dmap-based observations."""
    if type == "occupancy":
        if cfg.encoder_conv_map_occupancy_architecture.startswith("resnet"):
            return Resnet3DEncoder(cfg, obs_space, type)
        else:
            raise NotImplementedError(f"Unknown convolutional architecture {cfg.encoder_conv_map_occupancy_architecture}")