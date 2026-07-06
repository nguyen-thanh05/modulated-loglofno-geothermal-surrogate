import torch.nn as nn
from neuralop.models import UNO


class UNOWrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_layers,
        uno_out_channels,
        uno_n_modes,
        uno_scalings,
        lifting_channels=256,
        projection_channels=256,
        horizontal_skips_map=None,
        channel_mlp_skip="linear",
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.uno = UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            horizontal_skips_map=horizontal_skips_map,
            channel_mlp_skip=channel_mlp_skip,
            **kwargs,
        )

    def forward(self, x):
        return self.uno(x) + x[:, :self.out_channels]
