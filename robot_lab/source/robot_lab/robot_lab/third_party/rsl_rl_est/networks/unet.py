import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation


def _norm2d(norm: str, c: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(c)
    if norm == "group":
        g = 32 if c >= 32 else max(1, c // 4)
        return nn.GroupNorm(g, c)
    raise ValueError(f"Unsupported norm: {norm}")


class ConvBlock(nn.Module):
    """(Conv -> Norm -> Act) * 2 with configurable kernel/padding/dilation"""

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        norm: str = "none",
        activation: str = "elu",
        kernel_size: int = 3,
        padding: int | None = None,
        dilation: int = 1,
    ):
        super().__init__()
        act = resolve_nn_activation(activation)

        if padding is None:
            # "same" for odd kernel
            padding = ((kernel_size - 1) // 2) * dilation

        use_bias = (str(norm).lower() == "none")

        self.net = nn.Sequential(
            nn.Conv2d(
                cin, cout,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=use_bias,
            ),
            _norm2d(norm, cout),
            act,
            nn.Conv2d(
                cout, cout,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=use_bias,
            ),
            _norm2d(norm, cout),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool -> ConvBlock"""

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        norm: str = "none",
        activation: str = "elu",
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(
            cin, cout,
            norm=norm,
            activation=activation,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample -> concat skip -> ConvBlock"""

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        norm: str = "none",
        activation: str = "elu",
        kernel_size: int = 3,
        dilation: int = 1,
        mode: str = "bilinear",
    ):
        super().__init__()
        self.mode = mode
        self.conv = ConvBlock(
            cin, cout,
            norm=norm,
            activation=activation,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode=self.mode,
            align_corners=False if self.mode in ["bilinear", "bicubic"] else None,
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    in : [B, C, H, W]
    out: [B, out_channels, H, W]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        norm: str = "none",
        activation: str = "elu",
        upsample_mode: str = "bilinear",
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.inc = ConvBlock(
            in_channels, base_channels,
            norm=norm, activation=activation,
            kernel_size=kernel_size, dilation=dilation,
        )

        # encoder channels
        enc_ch = [base_channels * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(
                Down(
                    enc_ch[i], enc_ch[i + 1],
                    norm=norm, activation=activation,
                    kernel_size=kernel_size, dilation=dilation,
                )
            )

        # bottleneck: one more down (doubling channels)
        self.bottleneck = Down(
            enc_ch[-1], enc_ch[-1] * 2,
            norm=norm, activation=activation,
            kernel_size=kernel_size, dilation=dilation,
        )
        bott_ch = enc_ch[-1] * 2

        # decoder (match skips)
        self.ups = nn.ModuleList()
        dec_in = bott_ch
        for i in reversed(range(depth)):
            skip_ch = enc_ch[i]
            self.ups.append(
                Up(
                    cin=dec_in + skip_ch,
                    cout=skip_ch,
                    norm=norm, activation=activation,
                    kernel_size=kernel_size, dilation=dilation,
                    mode=upsample_mode,
                )
            )
            dec_in = skip_ch

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        return self.outc(x)