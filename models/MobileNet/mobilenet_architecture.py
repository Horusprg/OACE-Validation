import torch
import torch.nn as nn
from typing import Optional, Callable, List
import random
from pydantic import BaseModel

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
        resolution_multiplier: float = 1.0,
        dropout_rate: float = 0.0,
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

class MobilenetParams(BaseModel):
    num_classes: int = 10
    min_channels: int = 32
    max_channels: int = 1024
    dropout_rate: float = 0.0
    num_layers: int = 13
    batch_norm: bool = True

def generate_mobilenet_architecture(params: MobilenetParams) -> MobileNet:
    """
    Gera uma instância MobileNet com parâmetros simplificados.
    
    Args:
        params: Parâmetros da arquitetura contendo:
            - num_classes: Número de classes de saída
            - min_channels: Número mínimo de canais
            - max_channels: Número máximo de canais
            - dropout_rate: Taxa de dropout
            - num_layers: Número de camadas
            - batch_norm: Se deve usar batch normalization
    """
    # Calcula o número de canais para cada camada
    channels = []
    step = (params.max_channels - params.min_channels) / (params.num_layers - 1)
    for i in range(params.num_layers):
        channels.append(int(params.min_channels + step * i))
    
    # Configura as camadas com stride alternado
    conv_configs = []
    for i, c in enumerate(channels):
        stride = 2 if i in [2, 4, 6, -2] else 1  # Camadas com stride 2 em posições específicas
        conv_configs.append((c, stride))

    print(f"MobileNet: channels={channels}, dropout={params.dropout_rate}, num_layers={params.num_layers}")
    
    return MobileNet(
        num_classes=params.num_classes,
        conv_configs=conv_configs,
        dropout_rate=params.dropout_rate,
        batch_norm=params.batch_norm,
    )

if __name__ == "__main__":
    # Teste da geração da arquitetura
    print("\nTestando geração da arquitetura MobileNet:")
    
    # Teste 1: Configuração padrão
    params_padrao = MobilenetParams()
    modelo_padrao = generate_mobilenet_architecture(params_padrao)
    print("\nTeste 1 - Configuração padrão:")
    print(f"Número de parâmetros: {sum(p.numel() for p in modelo_padrao.parameters())}")
    
    # Teste 2: Configuração personalizada
    params_custom = MobilenetParams(
        num_classes=5,
        min_channels=16,
        max_channels=512,
        dropout_rate=0.3,
        num_layers=8,
        batch_norm=True
    )
    modelo_custom = generate_mobilenet_architecture(params_custom)
    print("\nTeste 2 - Configuração personalizada:")
    print(f"Número de parâmetros: {sum(p.numel() for p in modelo_custom.parameters())}")
    
    # Teste de forward pass
    print("\nTestando forward pass:")
    x_teste = torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=224, width=224
    
    try:
        saida_padrao = modelo_padrao(x_teste)
        print(f"Shape da saída (configuração padrão): {saida_padrao.shape}")
        
        saida_custom = modelo_custom(x_teste)
        print(f"Shape da saída (configuração personalizada): {saida_custom.shape}")
        print("\nTestes concluídos com sucesso!")
    except Exception as e:
        print(f"\nErro durante o teste: {str(e)}")