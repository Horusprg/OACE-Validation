import torch
import torch.nn as nn
from typing import List, Union
import random


class CNNBlock(nn.Module):
    """
    Bloco Convolucional Genérico.

    Consiste em uma sequência de Convolução 2D, Normalização em Lote (opcional),
    Função de Ativação e Dropout (opcional).
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
        dropout_rate: float = 0.0,
    ):
        """
        Inicializa o bloco convolucional.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            kernel_size: Tamanho do kernel da convolução (padrão: 3).
            stride: Stride da convolução (padrão: 1).
            padding: Padding da convolução (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            batch_norm: Se True, aplica normalização em lote.
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
        """
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not batch_norm,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos da convolução.

        Usa inicialização Kaiming para ativações ReLU-like (ReLU, LeakyReLU, ELU, SELU)
        e Xavier para outras. Bias é zerado se presente.

        Args:
            activation_fn: Função de ativação usada na camada.
        """
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco convolucional.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels, altura', largura']).
        """
        return self.block(x)


class GenericCNN(nn.Module):
    """
    Implementação de uma Rede Neural Convolucional (CNN) genérica.

    Permite a construção de uma rede com múltiplos estágios convolucionais,
    seguidos por camadas densas para classificação.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block_channels: List[int],
        kernel_sizes: Union[int, List[int]] = 3,
        strides: Union[int, List[int]] = 1,
        paddings: Union[int, List[int]] = 1,
        pooling_type: Union[str, None] = "max",
        pooling_kernel_size: int = 2,
        pooling_stride: int = 2,
        activation_fn: nn.Module = nn.ReLU,
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        fc_layers: List[int] = None,
    ):
        """
        Inicializa a CNN genérica.

        Args:
            in_channels: Número de canais de entrada (ex.: 3 para RGB).
            num_classes: Número de classes de saída.
            block_channels: Lista de canais de saída para cada estágio convolucional.
            kernel_sizes: Tamanho do kernel para cada camada convolucional. Pode ser um int
                          para todas as camadas ou uma lista de ints.
            strides: Stride para cada camada convolucional. Pode ser um int para todas
                     as camadas ou uma lista de ints.
            paddings: Padding para cada camada convolucional. Pode ser um int para todas
                      as camadas ou uma lista de ints.
            pooling_type: Tipo de pooling ('max', 'avg', ou None para não usar pooling).
            pooling_kernel_size: Tamanho do kernel para as camadas de pooling.
            pooling_stride: Stride para as camadas de pooling.
            activation_fn: Função de ativação (padrão: nn.ReLU).
            batch_norm: Se True, aplica normalização em lote.
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            fc_layers: Lista de tamanhos das camadas densas após as camadas convolucionais.
                       Se None ou vazia, nenhuma camada densa adicional é usada antes da camada final.

        Raises:
            ValueError: Se os parâmetros de listas (block_channels, kernel_sizes, etc.)
                        não forem consistentes ou inválidos.
        """
        super().__init__()

        if not isinstance(block_channels, list) or not all(
            isinstance(x, int) for x in block_channels
        ):
            raise ValueError("block_channels deve ser uma lista de inteiros.")
        if fc_layers is not None and (
            not isinstance(fc_layers, list)
            or not all(isinstance(x, int) for x in fc_layers)
        ):
            raise ValueError("fc_layers deve ser uma lista de inteiros ou None.")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1.")

        num_conv_blocks = len(block_channels)

        # Trata kernel_sizes, strides e paddings para que sejam listas
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_conv_blocks
        if isinstance(strides, int):
            strides = [strides] * num_conv_blocks
        if isinstance(paddings, int):
            paddings = [paddings] * num_conv_blocks

        if not (
            len(kernel_sizes) == num_conv_blocks
            and len(strides) == num_conv_blocks
            and len(paddings) == num_conv_blocks
        ):
            raise ValueError(
                "As listas kernel_sizes, strides e paddings devem ter o mesmo comprimento que block_channels."
            )

        layers = []
        current_in_channels = in_channels

        for i, out_c in enumerate(block_channels):
            layers.append(
                CNNBlock(
                    current_in_channels,
                    out_c,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation_fn=activation_fn,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                )
            )
            current_in_channels = out_c
            if pooling_type == "max":
                layers.append(
                    nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)
                )
            elif pooling_type == "avg":
                layers.append(
                    nn.AvgPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)
                )

        self.features = nn.Sequential(*layers)

        # Camada de pooling global antes das camadas densas
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Camadas densas (Fully Connected)
        fc_layers_list = []
        # Para calcular o tamanho da entrada para a primeira camada FC
        # Precisamos de um forward pass temporário para determinar o número de features após as convoluções
        # Este é um truque comum para determinar o tamanho de entrada para as camadas FC dinamicamente
        with torch.no_grad():
            dummy_input = torch.randn(
                1, in_channels, 32, 32
            )  # Assumindo 32x32 para um tamanho inicial
            dummy_output = self.features(dummy_input)
            dummy_output = self.avgpool(dummy_output)
            flattened_features = dummy_output.view(dummy_output.size(0), -1).shape[1]

        prev_fc_size = flattened_features
        if fc_layers:
            for fc_size in fc_layers:
                fc_layers_list.append(nn.Linear(prev_fc_size, fc_size))
                if batch_norm:  # Opcional: Batch norm em camadas densas
                    fc_layers_list.append(nn.BatchNorm1d(fc_size))
                fc_layers_list.append(activation_fn())
                if dropout_rate > 0:
                    fc_layers_list.append(nn.Dropout(dropout_rate))
                prev_fc_size = fc_size

        self.classifier = nn.Sequential(*fc_layers_list)
        self.output_layer = nn.Linear(prev_fc_size, num_classes)
        self._initialize_fc_weights()

    def _initialize_fc_weights(self) -> None:
        """
        Inicializa os pesos das camadas Fully Connected.
        """
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da CNN genérica.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Achatando o tensor
        x = self.classifier(x)
        return self.output_layer(x)


def generate_cnn_architecture(
    in_channels: int = 3,
    num_classes: int = 10,
    num_conv_blocks: int = 3,
    min_channels_per_block: int = 32,
    max_channels_per_block: int = 256,
    kernel_size_choice: Union[int, List[int], str] = "random",
    stride_choice: Union[int, List[int], str] = 1,
    padding_choice: Union[int, List[int], str] = "same",
    pooling_type_choice: str = "max",
    pooling_kernel_size: int = 2,
    pooling_stride: int = 2,
    activation_function_choice: str = "relu",
    dropout_rate: float = 0.0,
    batch_norm: bool = True,
    num_fc_layers: int = 1,
    min_fc_neurons: int = 64,
    max_fc_neurons: int = 512,
) -> GenericCNN:
    """
    Gera uma instância de uma CNN genérica com configuração randomizada.

    Args:
        in_channels: Número de canais de entrada (padrão: 3).
        num_classes: Número de classes de saída (padrão: 10).
        num_conv_blocks: Número de blocos convolucionais na rede (padrão: 3).
        min_channels_per_block: Mínimo de canais de saída por bloco convolucional.
        max_channels_per_block: Máximo de canais de saída por bloco convolucional.
        kernel_size_choice: Escolha para o tamanho do kernel ('random', 'fixed', ou uma lista).
                            'random' escolhe entre 3, 5, 7. 'fixed' usa 3.
        stride_choice: Escolha para o stride ('random', 'fixed', ou uma lista).
                       'random' escolhe entre 1, 2. 'fixed' usa 1.
        padding_choice: Escolha para o padding ('same', 'valid', ou um int/lista).
                        'same' calcula padding para manter o tamanho. 'valid' usa 0.
        pooling_type_choice: Tipo de pooling ('max', 'avg', 'none'; padrão: 'max').
        pooling_kernel_size: Tamanho do kernel para as camadas de pooling (padrão: 2).
        pooling_stride: Stride para as camadas de pooling (padrão: 2).
        activation_function_choice: Função de ativação ('relu', 'leaky_relu', 'elu', 'selu', 'gelu'; padrão: 'relu').
        dropout_rate: Taxa de dropout (0-1, padrão: 0).
        batch_norm: Se True, aplica normalização em lote (padrão: True).
        num_fc_layers: Número de camadas densas após as convoluções (padrão: 1).
        min_fc_neurons: Mínimo de neurônios por camada densa.
        max_fc_neurons: Máximo de neurônios por camada densa.

    Returns:
        GenericCNN: Instância configurada da CNN genérica.

    Raises:
        ValueError: Se dropout_rate, intervalos ou escolhas de parâmetros forem inválidos.
    """
    if not (0 <= dropout_rate <= 1):
        raise ValueError("dropout_rate deve estar entre 0 e 1.")
    if (
        min_channels_per_block > max_channels_per_block
        or min_fc_neurons > max_fc_neurons
    ):
        raise ValueError("Intervalos inválidos para canais ou neurônios FC.")

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    activation_fn = activation_map.get(activation_function_choice.lower(), nn.ReLU)

    block_channels = [
        random.randint(min_channels_per_block, max_channels_per_block)
        for _ in range(num_conv_blocks)
    ]

    # Configuração de kernel_sizes
    if isinstance(kernel_size_choice, str):
        if kernel_size_choice.lower() == "random":
            kernel_sizes = [random.choice([3, 5, 7]) for _ in range(num_conv_blocks)]
        elif kernel_size_choice.lower() == "fixed":
            kernel_sizes = [3] * num_conv_blocks
        else:
            raise ValueError(
                "kernel_size_choice inválido. Use 'random', 'fixed' ou uma lista de ints."
            )
    else:  # Assume que é uma lista de ints ou um int
        kernel_sizes = kernel_size_choice

    # Configuração de strides
    if isinstance(stride_choice, str):
        if stride_choice.lower() == "random":
            strides = [random.choice([1, 2]) for _ in range(num_conv_blocks)]
        elif stride_choice.lower() == "fixed":
            strides = [1] * num_conv_blocks
        else:
            raise ValueError(
                "stride_choice inválido. Use 'random', 'fixed' ou uma lista de ints."
            )
    else:  # Assume que é uma lista de ints ou um int
        strides = stride_choice

    # Configuração de paddings
    paddings = []
    if isinstance(padding_choice, str):
        if padding_choice.lower() == "same":
            # Para 'same' padding, padding = (kernel_size - 1) // 2
            paddings = [(k - 1) // 2 for k in kernel_sizes]
        elif padding_choice.lower() == "valid":
            paddings = [0] * num_conv_blocks
        else:
            raise ValueError(
                "padding_choice inválido. Use 'same', 'valid' ou um int/lista de ints."
            )
    else:  # Assume que é uma lista de ints ou um int
        paddings = padding_choice

    # Configuração das camadas Fully Connected
    fc_layers = []
    for _ in range(num_fc_layers):
        fc_layers.append(random.randint(min_fc_neurons, max_fc_neurons))

    print(
        f"CNN Gerada: Canais={block_channels}, Kernels={kernel_sizes}, Strides={strides}, Paddings={paddings}, "
        f"Pooling={pooling_type_choice}, Ativação={activation_function_choice}, Dropout={dropout_rate}, "
        f"BatchNorm={batch_norm}, FC Layers={fc_layers}"
    )

    return GenericCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        block_channels=block_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        pooling_type=pooling_type_choice,
        pooling_kernel_size=pooling_kernel_size,
        pooling_stride=pooling_stride,
        activation_fn=activation_fn,
        batch_norm=batch_norm,
        dropout_rate=dropout_rate,
        fc_layers=fc_layers,
    )


if __name__ == "__main__":
    print("--- Testando a arquitetura CNN Genérica ---")

    # Teste 1: CNN básica com 3 blocos convolucionais
    print("\n--- Teste 1: CNN Básica ---")
    try:
        model1 = generate_cnn_architecture(
            in_channels=3,
            num_classes=10,
            num_conv_blocks=3,
            min_channels_per_block=16,
            max_channels_per_block=32,
            kernel_size_choice=3,  # Kernel size fixo
            stride_choice=1,  # Stride fixo
            padding_choice="same",  # Padding 'same'
            pooling_type_choice="max",
            dropout_rate=0.1,
            batch_norm=True,
            num_fc_layers=1,
            min_fc_neurons=128,
            max_fc_neurons=128,
        )
        x1 = torch.randn(1, 3, 32, 32)
        output1 = model1(x1)
        print(f"Saída CNN Básica: {output1.shape}")
        assert output1.shape == torch.Size(
            [1, 10]
        ), "Erro no formato de saída da CNN Básica."
        print("CNN Básica testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar CNN Básica: {e}")

    # Teste 2: CNN com mais blocos, diferentes tamanhos de kernel e sem pooling
    print("\n--- Teste 2: CNN Complexa (sem pooling, kernels variados) ---")
    try:
        model2 = generate_cnn_architecture(
            in_channels=3,
            num_classes=100,
            num_conv_blocks=5,
            min_channels_per_block=64,
            max_channels_per_block=128,
            kernel_size_choice="random",  # Kernel size aleatório
            stride_choice=1,
            padding_choice="same",
            pooling_type_choice="none",  # Sem pooling
            activation_function_choice="leaky_relu",
            dropout_rate=0.2,
            batch_norm=True,
            num_fc_layers=2,
            min_fc_neurons=256,
            max_fc_neurons=512,
        )
        x2 = torch.randn(1, 3, 64, 64)  # Entrada maior para rede mais profunda
        output2 = model2(x2)
        print(f"Saída CNN Complexa: {output2.shape}")
        assert output2.shape == torch.Size(
            [1, 100]
        ), "Erro no formato de saída da CNN Complexa."
        print("CNN Complexa testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar CNN Complexa: {e}")

    # Teste 3: CNN com AvgPooling e menos camadas FC
    print("\n--- Teste 3: CNN com AvgPooling e FC mínima ---")
    try:
        model3 = generate_cnn_architecture(
            in_channels=3,
            num_classes=5,
            num_conv_blocks=2,
            min_channels_per_block=8,
            max_channels_per_block=16,
            pooling_type_choice="avg",
            dropout_rate=0.0,
            batch_norm=False,  # Sem Batch Norm
            num_fc_layers=0,  # Nenhuma camada FC extra
        )
        x3 = torch.randn(1, 3, 28, 28)
        output3 = model3(x3)
        print(f"Saída CNN AvgPooling: {output3.shape}")
        assert output3.shape == torch.Size(
            [1, 5]
        ), "Erro no formato de saída da CNN AvgPooling."
        print("CNN AvgPooling testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar CNN AvgPooling: {e}")
