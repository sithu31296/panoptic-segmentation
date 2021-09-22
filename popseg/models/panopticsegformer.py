import torch
from torch import nn, Tensor
from backbones import PVTv2


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PanopticSegFormer(nn.Module):
    def __init__(self, variant: str = 'B1'):
        super().__init__()
        self.backbone = PVTv2(variant)
        self.linear_c3 = MLP(self.backbone.embed_dims[1], 256)
        self.linear_c4 = MLP(self.backbone.embed_dims[2], 256)
        self.linear_c5 = MLP(self.backbone.embed_dims[3], 256)

    def forward(self, x: Tensor) -> Tensor:
        c3, c4, c5 = self.backbone(x)
        c3, c4, c5 = self.linear_c3(c3), self.linear_c4(c4), self.linear_c5(c5)
        out = torch.cat([c5, c4, c3], dim=1)
        print(out.shape)

if __name__ == '__main__':
    model = PanopticSegFormer('B1')
    x = torch.randn(2, 3, 224, 224)
    model(x)