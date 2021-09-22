import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .modules import Conv, SeparableConv
from .backbones import ResNet


class ASPP(nn.Module):
    def __init__(self, c1, c2, drop_rate=0.1):
        super().__init__()
        ratios = [1, 6, 12, 18]
        self.blocks = nn.ModuleList([
            Conv(c1, c2, 1 if ratio==1 else 3, 1, 0 if ratio==1 else ratio, ratio)
        for ratio in ratios])

        self.blocks.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, c2, 1)
        ))
        self.conv = Conv(c2 * (len(ratios) + 1), c2, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        contexts = []
        for blk in self.blocks:
            contexts.append(F.interpolate(blk(x), x.shape[2:], mode='bilinear', align_corners=False))

        x = self.conv(torch.cat(contexts, dim=1))
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, backbone_channels, aspp_out_channel=256, decoder_channel=256, low_level_channels=[64, 32]):
        super().__init__()
        self.aspp = ASPP(backbone_channels[-1], aspp_out_channel)
        self.conv = Conv(aspp_out_channel, aspp_out_channel, 1)

        self.project8 = Conv(backbone_channels[1], low_level_channels[0], 1)
        self.fuse8 = SeparableConv(aspp_out_channel + low_level_channels[0], decoder_channel, 5, 1, 2)

        self.project4 = Conv(backbone_channels[0], low_level_channels[1], 1)
        self.fuse4 = SeparableConv(decoder_channel + low_level_channels[1], decoder_channel, 5, 1, 2)

    def forward(self, features: list) -> Tensor:
        x = self.aspp(features[-1])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        feat8 = self.project8(features[1])
        x = self.fuse8(torch.cat([x, feat8], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        feat4 = self.project4(features[0])
        x = self.fuse4(torch.cat([x, feat4], dim=1))
        return x


class SemanticHead(nn.Module):
    def __init__(self, decoder_channel, head_channel, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv(decoder_channel, head_channel, 5, 1, 2),
            nn.Conv2d(head_channel, num_classes, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class InstanceHead(nn.Module):
    def __init__(self, decoder_channel, head_channel):
        super().__init__()
        self.center_conv = nn.Sequential(
            SeparableConv(decoder_channel, head_channel, 5, 1, 2),
            nn.Conv2d(head_channel, 1, 1)
        )
        self.offset_conv = nn.Sequential(
            SeparableConv(decoder_channel, head_channel, 5, 1, 2),
            nn.Conv2d(head_channel, 2, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.center_conv(x), self.offset_conv(x)


class PanopticDeepLab(nn.Module):
    def __init__(self, variant: str = '50', num_classes: int = 19):
        super().__init__()
        self.backbone = ResNet(variant)
        backbone_channels = [256, 512, 1024]
        
        self.semantic_decoder = Decoder(backbone_channels, 256, 256, [64, 32])
        self.instance_decoder = Decoder(backbone_channels, 256, 128, [32, 16])

        self.semantic_head = SemanticHead(256, 256, num_classes)
        self.instance_head = InstanceHead(128, 32)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)[:-1]
        semantic = self.semantic_decoder(features)
        instance = self.instance_decoder(features)

        semantic = self.semantic_head(semantic)
        center, offset = self.instance_head(instance)

        semantic = F.interpolate(semantic, x.shape[-2:], mode='bilinear', align_corners=False)
        center = F.interpolate(center, x.shape[-2:], mode='bilinear', align_corners=False)

        scale = x.shape[-2] // offset.shape[-2]
        offset = F.interpolate(offset, x.shape[-2:], mode='bilinear', align_corners=False)
        offset *= scale
        
        return semantic, center, offset


if __name__ == '__main__':
    model = PanopticDeepLab('50')
    x = torch.randn(2, 3, 224, 224)
    semantic, center, offset = model(x)
    print(semantic.shape)
    print(center.shape)
    print(offset.shape)