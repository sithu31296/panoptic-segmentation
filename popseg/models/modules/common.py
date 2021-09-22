import torch
from torch import nn, Tensor


class Conv(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class SeparableConv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__()
        self.dw_conv = Conv(c1, c1, k, s, p, g=g)
        self.pw_conv = Conv(c1, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x