import torch
import torch.nn as nn
from typing import List, Union, Optional
import random


class VGGBlock(nn.Module):
    """
    Bloco Convolucional VGG.
    
    Consiste em convoluções 3x3 seguidas por normalização em lote (opcional),
    ativação ReLU e dropout (opcional).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Inicializa o bloco VGG.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            batch_norm: Se True, aplica normalização em lote.
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
        """
        super().__init__()
        layers = []
        
        # Convolução 3x3 com padding=1 (padrão VGG)
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not batch_norm,
            )
        )
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Inicializa os pesos da convolução usando inicialização Kaiming.
        """
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco VGG.

        Args:
            x: Tensor de entrada.

        Returns:
            Tensor processado.
        """
        return self.block(x)


class VGG(nn.Module):
    """
    Implementação da arquitetura VGG.
    
    A VGG é caracterizada por usar apenas convoluções 3x3 empilhadas,
    seguidas por max pooling 2x2.
    """

    def __init__(
        self,
        conv_layers_config: List[Union[int, str]],
        fc_layers: List[int],
        num_classes: int,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
        batch_norm: bool = False,
        weight_init_fn: Optional[str] = None,
    ):
        """
        Inicializa a arquitetura VGG.

        Args:
            conv_layers_config: Lista definindo a sequência de camadas convolucionais.
                               Números inteiros representam canais de saída para convoluções 3x3.
                               'M' representa max pooling 2x2.
            fc_layers: Lista de tamanhos das camadas totalmente conectadas.
            num_classes: Número de classes de saída.
            in_channels: Número de canais de entrada (padrão: 3 para RGB).
            dropout_rate: Taxa de dropout nas camadas FC (padrão: 0.5).
            batch_norm: Se True, aplica normalização em lote.
            weight_init_fn: Tipo de inicialização de pesos ('kaiming', 'xavier', ou None).

        Raises:
            ValueError: Se os parâmetros forem inválidos.
        """
        super().__init__()

        if not isinstance(conv_layers_config, list):
            raise ValueError("conv_layers_config deve ser uma lista.")
        if not isinstance(fc_layers, list) or not all(isinstance(x, int) for x in fc_layers):
            raise ValueError("fc_layers deve ser uma lista de inteiros.")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1.")

        self.conv_layers_config = conv_layers_config
        self.fc_layers = fc_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Construir camadas convolucionais
        self.features = self._make_conv_layers(in_channels)
        
        # Pooling adaptativo antes das camadas FC
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Construir classificador (camadas FC)
        self.classifier = self._make_classifier()
        
        # Inicializar pesos
        if weight_init_fn:
            self._initialize_weights(weight_init_fn)

    def _make_conv_layers(self, in_channels: int) -> nn.Sequential:
        """
        Constrói as camadas convolucionais baseadas na configuração.

        Args:
            in_channels: Número de canais de entrada.

        Returns:
            nn.Sequential: Sequência de camadas convolucionais e pooling.
        """
        layers = []
        current_channels = in_channels

        for config in self.conv_layers_config:
            if config == 'M':
                # Max pooling 2x2
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # Camada convolucional
                layers.append(
                    VGGBlock(
                        current_channels,
                        config,
                        batch_norm=self.batch_norm,
                        dropout_rate=0.0,  # Dropout geralmente não usado nas conv layers da VGG
                    )
                )
                current_channels = config

        return nn.Sequential(*layers)

    def _make_classifier(self) -> nn.Sequential:
        """
        Constrói o classificador (camadas totalmente conectadas).

        Returns:
            nn.Sequential: Sequência de camadas FC.
        """
        # Calcular número de features após convoluções
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Tamanho padrão ImageNet
            dummy_features = self.features(dummy_input)
            dummy_features = self.avgpool(dummy_features)
            num_features = dummy_features.view(dummy_features.size(0), -1).shape[1]

        layers = []
        prev_size = num_features

        # Camadas FC intermediárias
        for fc_size in self.fc_layers:
            layers.append(nn.Linear(prev_size, fc_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.dropout_rate))
            prev_size = fc_size

        # Camada de saída
        layers.append(nn.Linear(prev_size, self.num_classes))

        return nn.Sequential(*layers)

    def _initialize_weights(self, weight_init_fn: str) -> None:
        """
        Inicializa os pesos da rede.

        Args:
            weight_init_fn: Tipo de inicialização ('kaiming' ou 'xavier').
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if weight_init_fn.lower() == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif weight_init_fn.lower() == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if weight_init_fn.lower() == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif weight_init_fn.lower() == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da VGG.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def generate_vgg_architecture(
    num_classes: int = 1000,
    in_channels: int = 3,
    dropout_rate: float = 0.5,
    batch_norm: bool = False,
    weight_init_fn: str = "kaiming",
    # Parâmetros para randomização
    min_conv_layers_per_block: int = 1,
    max_conv_layers_per_block: int = 3,
    min_channels: int = 64,
    max_channels: int = 512,
    num_conv_blocks: int = 5,
    min_fc_neurons: int = 1024,
    max_fc_neurons: int = 4096,
    num_fc_layers: int = 2,
) -> VGG:
    """
    Gera uma instância da arquitetura VGG com configuração randomizada.

    Args:
        num_classes: Número de classes de saída.
        in_channels: Número de canais de entrada.
        dropout_rate: Taxa de dropout nas camadas FC.
        batch_norm: Se True, aplica normalização em lote.
        weight_init_fn: Tipo de inicialização de pesos.
        min_conv_layers_per_block: Mínimo de camadas conv por bloco.
        max_conv_layers_per_block: Máximo de camadas conv por bloco.
        min_channels: Mínimo de canais por camada conv.
        max_channels: Máximo de canais por camada conv.
        num_conv_blocks: Número de blocos convolucionais.
        min_fc_neurons: Mínimo de neurônios por camada FC.
        max_fc_neurons: Máximo de neurônios por camada FC.
        num_fc_layers: Número de camadas FC.

    Returns:
        VGG: Instância configurada da VGG.

    Raises:
        ValueError: Se os parâmetros forem inválidos.
    """
    if not 0 <= dropout_rate <= 1:
        raise ValueError("dropout_rate deve estar entre 0 e 1.")
    if min_channels > max_channels or min_fc_neurons > max_fc_neurons:
        raise ValueError("Intervalos inválidos para canais ou neurônios FC.")
    if min_conv_layers_per_block > max_conv_layers_per_block:
        raise ValueError("Intervalo inválido para camadas por bloco.")

    # Gerar configuração de camadas convolucionais randomizada
    conv_layers_config = []
    current_channels = min_channels
    
    for block in range(num_conv_blocks):
        # Número de camadas convolucionais neste bloco
        num_layers_in_block = random.randint(min_conv_layers_per_block, max_conv_layers_per_block)
        
        # Adicionar camadas convolucionais
        for _ in range(num_layers_in_block):
            conv_layers_config.append(current_channels)
        
        # Adicionar max pooling após cada bloco
        conv_layers_config.append('M')
        
        # Aumentar número de canais (padrão VGG: dobrar até max_channels)
        current_channels = min(current_channels * 2, max_channels)
    
    # Configurar camadas FC randomizadas
    fc_layers = [
        random.randint(min_fc_neurons, max_fc_neurons)
        for _ in range(num_fc_layers)
    ]

    print(f"VGG Gerada: Conv Config={conv_layers_config}, FC Layers={fc_layers}, "
          f"Dropout={dropout_rate}, BatchNorm={batch_norm}, WeightInit={weight_init_fn}")

    return VGG(
        conv_layers_config=conv_layers_config,
        fc_layers=fc_layers,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        weight_init_fn=weight_init_fn,
    )


if __name__ == "__main__":
    print("--- Testando a arquitetura VGG ---")

    # Teste 1: VGG básica randomizada
    print("\n--- Teste 1: VGG Básica ---")
    try:
        model1 = generate_vgg_architecture(
            num_classes=10,
            num_conv_blocks=3,
            min_conv_layers_per_block=1,
            max_conv_layers_per_block=2,
            min_channels=32,
            max_channels=128,
            num_fc_layers=1,
            min_fc_neurons=256,
            max_fc_neurons=512,
            batch_norm=True,
            dropout_rate=0.3
        )
        x1 = torch.randn(1, 3, 224, 224)
        output1 = model1(x1)
        print(f"Saída VGG Básica: {output1.shape}")
        assert output1.shape == torch.Size([1, 10]), "Erro no formato de saída da VGG Básica."
        print("VGG Básica testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar VGG Básica: {e}")

    # Teste 2: VGG mais profunda
    print("\n--- Teste 2: VGG Profunda ---")
    try:
        model2 = generate_vgg_architecture(
            num_classes=100,
            num_conv_blocks=5,
            min_conv_layers_per_block=2,
            max_conv_layers_per_block=3,
            min_channels=64,
            max_channels=512,
            num_fc_layers=2,
            min_fc_neurons=1024,
            max_fc_neurons=4096,
            batch_norm=False,
            dropout_rate=0.5
        )
        x2 = torch.randn(1, 3, 224, 224)
        output2 = model2(x2)
        print(f"Saída VGG Profunda: {output2.shape}")
        assert output2.shape == torch.Size([1, 100]), "Erro no formato de saída da VGG Profunda."
        print("VGG Profunda testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar VGG Profunda: {e}")

    # Teste 3: VGG com entrada menor
    print("\n--- Teste 3: VGG com entrada 32x32 ---")
    try:
        model3 = generate_vgg_architecture(
            num_classes=5,
            num_conv_blocks=2,
            min_conv_layers_per_block=1,
            max_conv_layers_per_block=1,
            min_channels=16,
            max_channels=64,
            num_fc_layers=1,
            min_fc_neurons=128,
            max_fc_neurons=256,
            batch_norm=True,
            dropout_rate=0.2
        )
        x3 = torch.randn(1, 3, 32, 32)
        output3 = model3(x3)
        print(f"Saída VGG (32x32): {output3.shape}")
        assert output3.shape == torch.Size([1, 5]), "Erro no formato de saída da VGG."
        print("VGG com entrada 32x32 testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar VGG com entrada 32x32: {e}")

    print("\n--- Todos os testes concluídos ---")