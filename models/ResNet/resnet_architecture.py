import torch
import torch.nn as nn
from typing import List
import random


class ResNetInputLayer(nn.Module):
    """
    Camada inicial da ResNet.

    Aplica uma convolução inicial, normalização em lote (opcional) e função de ativação
    para processar a entrada da rede (ex.: imagens RGB do CIFAR-10).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        batch_norm: bool = True,
    ):
        """
        Inicializa a camada inicial.

        Args:
            in_channels: Número de canais de entrada (ex.: 3 para RGB).
            out_channels: Número de canais de saída.
            kernel_size: Tamanho do kernel da convolução (padrão: 3).
            stride: Stride da convolução (padrão: 1).
            padding: Padding da convolução (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            batch_norm: Se True, aplica normalização em lote.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_fn()
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos da convolução.

        Usa inicialização Kaiming para ativações ReLU-like (ReLU, LeakyReLU, ELU, SELU)
        e Xavier para outras. Bias é zerado se presente.

        Args:
            activation_fn: Função de ativação usada na camada.
        """
        if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada inicial.

        Aplica convolução, normalização em lote (se habilitada) e ativação.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels, altura', largura']).
        """
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return self.activation(x)


class BasicBlock(nn.Module):
    """
    Bloco residual básico para ResNet-18/34.

    Contém duas convoluções 3x3 com conexão residual, suportando dropout,
    normalização em lote e ativação customizada.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Inicializa o bloco residual básico.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            stride: Stride da primeira convolução (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            batch_norm: Se True, aplica normalização em lote.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not batch_norm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_fn()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=not batch_norm,
                ),
                (
                    nn.BatchNorm2d(out_channels * self.expansion)
                    if batch_norm
                    else nn.Identity()
                ),
            )
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos das convoluções.

        Usa Kaiming para ReLU-like e Xavier para outras ativações. Bias é zerado.

        Args:
            activation_fn: Função de ativação usada no bloco.
        """
        for conv in [self.conv1, self.conv2]:
            if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco residual.

        Aplica conv1 -> bn1 -> ativação -> dropout -> conv2 -> bn2, soma com
        o caminho residual (shortcut) e aplica ativação final.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels, altura', largura']).
        """
        identity = self.shortcut(x)
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x += identity
        return self.activation(x)


class Bottleneck(nn.Module):
    """
    Bloco Bottleneck para ResNet-50/101/152.

    Contém três convoluções (1x1, 3x3, 1x1) com conexão residual, suportando
    dropout, normalização em lote e ativação customizada.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Inicializa o bloco Bottleneck.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída (antes da expansão).
            stride: Stride da convolução 3x3 (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            batch_norm: Se True, aplica normalização em lote.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=not batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_fn()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=not batch_norm,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) if batch_norm else None
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=not batch_norm,
                ),
                (
                    nn.BatchNorm2d(out_channels * self.expansion)
                    if batch_norm
                    else nn.Identity()
                ),
            )
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos das convoluções.

        Usa Kaiming para ReLU-like e Xavier para outras ativações. Bias é zerado.

        Args:
            activation_fn: Função de ativação usada no bloco.
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco Bottleneck.

        Aplica conv1 -> bn1 -> act -> dropout -> conv2 -> bn2 -> act -> dropout ->
        conv3 -> bn3, soma com o caminho residual e aplica ativação final.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels * expansion, altura', largura']).
        """
        identity = self.shortcut(x)
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x += identity
        return self.activation(x)


class ResNetOutputLayer(nn.Module):
    """
    Camada de saída da ResNet.

    Aplica average pooling global e uma camada linear para mapear as features
    para o espaço de classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        """
        Inicializa a camada de saída.

        Args:
            in_channels: Número de canais de entrada (saída do último estágio).
            num_classes: Número de classes de saída.
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Inicializa os pesos da camada linear.

        Usa inicialização Xavier para os pesos e zera o bias.
        """
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada de saída.

        Aplica average pooling, achata o tensor e passa pela camada linear.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class ResNet(nn.Module):
    """
    Implementação da Residual Network (ResNet).

    Consiste em uma camada inicial, quatro estágios residuais (com blocos BasicBlock
    ou Bottleneck) e uma camada de saída. Suporta configurações flexíveis.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        block_type: nn.Module,
        layers: List[int],
        num_classes: int,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Inicializa a ResNet.

        Args:
            in_channels: Número de canais de entrada (ex.: 3 para RGB).
            base_channels: Número de canais base da camada inicial.
            block_type: Tipo de bloco residual (BasicBlock ou Bottleneck).
            layers: Lista com número de blocos por estágio (ex.: [2, 2, 2, 2]).
            num_classes: Número de classes de saída.
            activation_fn: Função de ativação (padrão: nn.ReLU).
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            batch_norm: Se True, aplica normalização em lote.

        Raises:
            ValueError: Se layers não for lista de inteiros, dropout_rate inválido,
                        ou block_type inválido.
        """
        super().__init__()
        if not isinstance(layers, list) or not all(isinstance(x, int) for x in layers):
            raise ValueError("layers deve ser uma lista de inteiros")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1")
        if block_type not in [BasicBlock, Bottleneck]:
            raise ValueError("block_type deve ser BasicBlock ou Bottleneck")

        self.in_channels = base_channels
        self.input_layer = ResNetInputLayer(
            in_channels, base_channels, 3, 1, 1, activation_fn, batch_norm
        )
        self.stage1 = self._make_stage(
            block_type,
            base_channels,
            layers[0],
            1,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.stage2 = self._make_stage(
            block_type,
            base_channels * 2,
            layers[1],
            2,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.stage3 = self._make_stage(
            block_type,
            base_channels * 4,
            layers[2],
            2,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.stage4 = self._make_stage(
            block_type,
            base_channels * 8,
            layers[3],
            2,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.output_layer = ResNetOutputLayer(
            base_channels * 8 * block_type.expansion, num_classes
        )

    def _make_stage(
        self,
        block_type: nn.Module,
        out_channels: int,
        num_blocks: int,
        stride: int,
        activation_fn: nn.Module,
        dropout_rate: float,
        batch_norm: bool,
    ) -> nn.Sequential:
        """
        Cria um estágio da ResNet com blocos residuais.

        Args:
            block_type: Tipo de bloco (BasicBlock ou Bottleneck).
            out_channels: Número de canais de saída (antes da expansão).
            num_blocks: Número de blocos no estágio.
            stride: Stride do primeiro bloco.
            activation_fn: Função de ativação.
            dropout_rate: Taxa de dropout.
            batch_norm: Se True, aplica normalização em lote.

        Returns:
            nn.Sequential contendo os blocos residuais.
        """
        layers = []
        layers.append(
            block_type(
                self.in_channels,
                out_channels,
                stride,
                activation_fn,
                dropout_rate,
                batch_norm,
            )
        )
        self.in_channels = out_channels * block_type.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block_type(
                    self.in_channels,
                    out_channels,
                    1,
                    activation_fn,
                    dropout_rate,
                    batch_norm,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da ResNet.

        Processa a entrada através da camada inicial, estágios residuais e camada
        de saída, retornando logits.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.input_layer(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.output_layer(x)


def generate_resnet_architecture(
    in_channels: int = 3,
    num_classes: int = 10,
    num_stages: int = 4,
    min_blocks_per_stage: int = 1,
    max_blocks_per_stage: int = 4,
    min_base_channels: int = 16,
    max_base_channels: int = 64,
    block_type_choice: str = "basic",
    activation_function_choice: str = "relu",
    dropout_rate: float = 0.0,
    batch_norm: bool = True,
) -> ResNet:
    """
    Gera uma instância da ResNet com configuração randomizada.

    Permite randomização de profundidade (blocos por estágio), tipo de bloco,
    largura (canais base) e configuração de ativação, dropout e batch norm.

    Args:
        in_channels: Número de canais de entrada (padrão: 3).
        num_classes: Número de classes de saída (padrão: 10).
        num_stages: Número de estágios residuais (padrão: 4).
        min_blocks_per_stage: Mínimo de blocos por estágio (padrão: 1).
        max_blocks_per_stage: Máximo de blocos por estágio (padrão: 4).
        min_base_channels: Mínimo de canais base (padrão: 16).
        max_base_channels: Máximo de canais base (padrão: 64).
        block_type_choice: Tipo de bloco ('basic', 'bottleneck', 'random'; padrão: 'basic').
        activation_function_choice: Função de ativação ('relu', 'leaky_relu', 'elu', 'selu', 'gelu'; padrão: 'relu').
        dropout_rate: Taxa de dropout (0-1, padrão: 0).
        batch_norm: Se True, aplica normalização em lote (padrão: True).

    Returns:
        ResNet: Instância configurada da ResNet.

    Raises:
        ValueError: Se dropout_rate ou intervalos de blocos/canais forem inválidos.
    """
    if not (0 <= dropout_rate <= 1):
        raise ValueError("dropout_rate deve estar entre 0 e 1")
    if (
        min_blocks_per_stage > max_blocks_per_stage
        or min_base_channels > max_base_channels
    ):
        raise ValueError("Intervalos inválidos para blocos ou canais")

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    block_map = {"basic": BasicBlock, "bottleneck": Bottleneck}

    activation_fn = activation_map.get(activation_function_choice.lower(), nn.ReLU)
    block_type = (
        random.choice([BasicBlock, Bottleneck])
        if block_type_choice.lower() == "random"
        else block_map.get(block_type_choice.lower(), BasicBlock)
    )

    layers = [
        random.randint(min_blocks_per_stage, max_blocks_per_stage)
        for _ in range(num_stages)
    ]
    base_channels = random.randint(min_base_channels, max_base_channels)

    print(
        f"ResNet: Bloco={block_type.__name__}, Camadas={layers}, Canais={base_channels}, Ativação={activation_function_choice}, Dropout={dropout_rate}, BatchNorm={batch_norm}"
    )

    return ResNet(
        in_channels,
        base_channels,
        block_type,
        layers,
        num_classes,
        activation_fn,
        dropout_rate,
        batch_norm,
    )


"""
def test_resnet_specific_layers(layers: List[int], model_name: str, block_type_choice: str = 'bottleneck', num_classes: int = 10, input_size: int = 224):
    #Testa a ResNet com um número específico de camadas.
    print(f"\n--- Testando {model_name} ---")
    try:
        model = ResNet(in_channels=3, base_channels=64, block_type=block_map.get(block_type_choice.lower(), BasicBlock),
                       layers=layers, num_classes=num_classes, activation_fn=nn.ReLU, dropout_rate=0.0, batch_norm=True)
        x = torch.randn(1, 3, input_size, input_size)
        output = model(x)
        print(f"Saída ({model_name}): {output.shape}")
        assert output.shape == torch.Size([1, num_classes]), f"Erro no formato de saída do {model_name}"
        print(f"{model_name} testado com sucesso.")
    except Exception as e:
        print(f"Erro ao testar {model_name}: {e}")

if __name__ == "__main__":
    print("--- Testando a arquitetura ResNet ---")

    # ... (seus testes anteriores) ...

    print("\n--- Testando arquiteturas ResNet específicas ---")

    block_map = {'basic': BasicBlock, 'bottleneck': Bottleneck}

    # Teste ResNet-50
    test_resnet_specific_layers(layers=[3, 4, 6, 3], model_name="ResNet-50")

    # Teste ResNet-101
    test_resnet_specific_layers(layers=[3, 4, 23, 3], model_name="ResNet-101")

    # Você pode adicionar um teste para ResNet-152 se desejar
    test_resnet_specific_layers(layers=[3, 8, 36, 3], model_name="ResNet-152")

    # Você também pode testar uma ResNet-50 com BasicBlock (embora não seja padrão)
    test_resnet_specific_layers(layers=[3, 4, 6, 3], model_name="ResNet-50 com BasicBlock", block_type_choice='basic')
    
    # Gerar ResNet
    model = generate_resnet_architecture(
        in_channels=3,
        num_classes=10,
        block_type_choice='basic',  # Testar com BasicBlock
        activation_function_choice='relu',
        min_base_channels=16,
        max_base_channels=16
    )

    # Entrada de teste
    x = torch.randn(1, 3, 32, 32)

    # Forward pass
    output = model(x)
    print(f"Saída ResNet: {output.shape}")  # Esperado: [1, 10]
"""
