import torch
import torch.nn as nn
from typing import Optional, Callable, List
import random

class DepthwiseSeparableConv(nn.Module):
    """
    Bloco de Convolução Separável em Profundidade (Depthwise + Pointwise).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation_fn=nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.activation = activation_fn()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MobileNet(nn.Module):
    """
    Implementação simplificada da MobileNet (v1).
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        resolution_multiplier: float = 1.0,
        dropout_rate: float = 0.0,
        weight_init_fn: Optional[Callable] = None,
        conv_configs: Optional[List] = None,
        batch_norm: bool = True,
    ):
        super().__init__()
        if conv_configs is None:
            conv_configs = [
                (32, 1),
                (64, 1),
                (128, 2),
                (128, 1),
                (256, 2),
                (256, 1),
                (512, 2),
                *[(512, 1)] * 5,
                (1024, 2),
                (1024, 1),
            ]
        conv_configs = [(int(c * width_multiplier), s) for c, s in conv_configs]
        layers = []
        in_channels = 3
        for out_channels, stride in conv_configs:
            layers.append(
                DepthwiseSeparableConv(
                    in_channels,
                    out_channels,
                    stride=stride,
                    activation_fn=nn.ReLU,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
        self.resolution_multiplier = resolution_multiplier
        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(self, x):
        if self.resolution_multiplier != 1.0:
            h, w = x.shape[2:]
            new_h = int(h * self.resolution_multiplier)
            new_w = int(w * self.resolution_multiplier)
            x = nn.functional.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def generate_mobilenet_architecture(
    num_classes: int = 10,
    width_multiplier: Optional[float] = None,
    resolution_multiplier: Optional[float] = None,
    dropout_rate: Optional[float] = None,
    weight_init_fn: Optional[Callable] = None,
    batch_norm: bool = True,
    randomize: bool = True,
) -> MobileNet:
    """
    Gera uma instância MobileNet com randomização de hiperparâmetros.
    """
    if randomize:
        width_multiplier = (
            width_multiplier
            if width_multiplier is not None
            else random.choice([0.5, 0.75, 1.0, 1.25])
        )
        resolution_multiplier = (
            resolution_multiplier
            if resolution_multiplier is not None
            else random.choice([0.5, 0.75, 1.0])
        )
        dropout_rate = (
            dropout_rate
            if dropout_rate is not None
            else random.choice([0.0, 0.1, 0.2, 0.3])
        )
    else:
        width_multiplier = width_multiplier or 1.0
        resolution_multiplier = resolution_multiplier or 1.0
        dropout_rate = dropout_rate or 0.0
    print(
        f"MobileNet: width_multiplier={width_multiplier}, resolution_multiplier={resolution_multiplier}, dropout={dropout_rate}"
    )
    return MobileNet(
        num_classes=num_classes,
        width_multiplier=width_multiplier,
        resolution_multiplier=resolution_multiplier,
        dropout_rate=dropout_rate,
        weight_init_fn=weight_init_fn,
        batch_norm=batch_norm,
    )

# if __name__ == "__main__":
#     model = generate_mobilenet_architecture(num_classes=10)
#     x = torch.randn(2, 3, 32, 32)
#     out = model(x)
#     print("Saída do modelo:", out.shape)