import torch
import torch.nn as nn
from typing import List, Union, Optional
import random
from pydantic import BaseModel


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
    Implementação da arquitetura VGG padronizada.
    
    A VGG é caracterizada por usar apenas convoluções 3x3 empilhadas,
    seguidas por max pooling 2x2.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: List[int] = [2, 2, 3, 3],
        num_classes: int = 10,
        activation_fn: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.5,
        batch_norm: bool = True,
    ):
        """
        Inicializa a arquitetura VGG padronizada.

        Args:
            in_channels: Número de canais de entrada (padrão: 3 para RGB).
            base_channels: Número de canais base para a primeira camada.
            layers: Lista com número de blocos por estágio.
            num_classes: Número de classes de saída.
            activation_fn: Função de ativação (padrão: ReLU).
            dropout_rate: Taxa de dropout nas camadas FC (padrão: 0.5).
            batch_norm: Se True, aplica normalização em lote.

        Raises:
            ValueError: Se os parâmetros forem inválidos.
        """
        super().__init__()

        if not isinstance(layers, list) or not all(isinstance(x, int) for x in layers):
            raise ValueError("layers deve ser uma lista de inteiros.")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1.")

        self.layers = layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Construir camadas convolucionais
        self.features = self._make_conv_layers(in_channels, base_channels)
        
        # Pooling adaptativo antes das camadas FC
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Construir classificador (camadas FC)
        self.classifier = self._make_classifier()
        
        # Inicializar pesos
        self._initialize_weights()

    def _make_conv_layers(self, in_channels: int, base_channels: int) -> nn.Sequential:
        """
        Constrói as camadas convolucionais baseadas na configuração.

        Args:
            in_channels: Número de canais de entrada.
            base_channels: Número de canais base.

        Returns:
            nn.Sequential: Sequência de camadas convolucionais e pooling.
        """
        layers = []
        current_channels = in_channels

        for stage_idx, num_blocks in enumerate(self.layers):
            # Número de canais para este estágio (dobra a cada estágio)
            out_channels = base_channels * (2 ** stage_idx)
            
            # Adiciona blocos convolucionais para este estágio
            for _ in range(num_blocks):
                layers.append(
                    VGGBlock(
                        current_channels,
                        out_channels,
                        batch_norm=self.batch_norm,
                        dropout_rate=0.0,  # Dropout geralmente não usado nas conv layers da VGG
                    )
                )
                current_channels = out_channels
            
            # Adiciona max pooling após cada estágio (exceto o último)
            if stage_idx < len(self.layers) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def _make_classifier(self) -> nn.Sequential:
        """
        Constrói o classificador (camadas totalmente conectadas).

        Returns:
            nn.Sequential: Sequência de camadas FC.
        """
        # Calcular número de features após convoluções
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)  # Tamanho CIFAR-10
            dummy_features = self.features(dummy_input)
            dummy_features = self.avgpool(dummy_features)
            num_features = dummy_features.view(dummy_features.size(0), -1).shape[1]

        # Configuração das camadas FC baseada no número de features
        if num_features > 2048:
            fc_layers = [2048, 1024]
        elif num_features > 1024:
            fc_layers = [1024, 512]
        else:
            fc_layers = [512, 256]

        layers = []
        prev_size = num_features

        # Camadas FC intermediárias
        for fc_size in fc_layers:
            layers.append(nn.Linear(prev_size, fc_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.dropout_rate))
            prev_size = fc_size

        # Camada de saída
        layers.append(nn.Linear(prev_size, self.num_classes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """
        Inicializa os pesos da rede usando inicialização Kaiming.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
    
class VGGParams(BaseModel):
    num_classes: int = 10
    min_channels: int = 32
    max_channels: int = 256
    dropout_rate: float = 0.5
    num_layers: int = 6
    batch_norm: bool = True

def adjust_vgg_params_for_memory(params: VGGParams) -> VGGParams:
    """
    Ajusta os parâmetros da VGG para evitar problemas de memória GPU.
    Aplica restrições específicas para VGG sem afetar outras arquiteturas.
    
    Args:
        params: Parâmetros originais da VGG
        
    Returns:
        VGGParams: Parâmetros ajustados para VGG
    """
    adjusted_params = params.copy()
    
    # Limita max_channels para VGG (mais conservador que MobileNet)
    if adjusted_params.max_channels > 512:
        print(f"⚠️  VGG: limitando max_channels de {adjusted_params.max_channels} para 512")
        adjusted_params.max_channels = 512
    
    # Limita min_channels para VGG
    if adjusted_params.min_channels > 64:
        print(f"⚠️  VGG: limitando min_channels de {adjusted_params.min_channels} para 64")
        adjusted_params.min_channels = 64
    
    # Limita num_layers para VGG (mais conservador)
    if adjusted_params.num_layers > 8:
        print(f"⚠️  VGG: limitando num_layers de {adjusted_params.num_layers} para 8")
        adjusted_params.num_layers = 8
    
    return adjusted_params

def generate_vgg_architecture(params: VGGParams) -> VGG:
    """
    Gera uma instância VGG com parâmetros simplificados seguindo o padrão MobileNet/CNN.
    
    Args:
        params: Parâmetros da arquitetura contendo:
            - num_classes: Número de classes de saída
            - min_channels: Número mínimo de canais base
            - max_channels: Número máximo de canais base
            - dropout_rate: Taxa de dropout
            - num_layers: Número de camadas convolucionais
            - batch_norm: Se deve usar batch normalization
    """
    # Ajusta parâmetros específicos para VGG (evita problemas de memória)
    adjusted_params = adjust_vgg_params_for_memory(params)
    
    # Calcula o número de canais base
    base_channels = adjusted_params.min_channels + (adjusted_params.max_channels - adjusted_params.min_channels) // 2
    
    # Configura a arquitetura VGG baseada no número de camadas
    if adjusted_params.num_layers < 4:
        layers = [1, 1, 1, 1]  # VGG mínima
    elif adjusted_params.num_layers == 4:
        layers = [1, 1, 1, 1]  # VGG-4
    elif adjusted_params.num_layers == 6:
        layers = [2, 2, 2, 2]  # VGG-8
    elif adjusted_params.num_layers == 8:
        layers = [2, 2, 3, 3]  # VGG-10
    else:
        # Para mais de 8 camadas, distribui os blocos
        total_blocks = min(adjusted_params.num_layers, 10)  # Limita a 10 blocos para VGG
        layers = [max(1, total_blocks // 4)] * 4
        # Distribui os blocos restantes
        remaining = total_blocks % 4
        for i in range(remaining):
            layers[i] += 1
    
    print(f"VGG: base_channels={base_channels}, layers={layers}, dropout={adjusted_params.dropout_rate}, num_layers={adjusted_params.num_layers}")
    
    return VGG(
        in_channels=3,
        base_channels=base_channels,
        layers=layers,
        num_classes=adjusted_params.num_classes,
        activation_fn=nn.ReLU(),
        dropout_rate=adjusted_params.dropout_rate,
        batch_norm=adjusted_params.batch_norm,
    )


if __name__ == "__main__":
    print("--- Testando a arquitetura VGG Padronizada ---")

    # Teste 1: VGG básica com configuração padrão
    print("\n--- Teste 1: VGG Básica ---")
    try:
        params_padrao = VGGParams()
        model1 = generate_vgg_architecture(params_padrao)
        x1 = torch.randn(2, 3, 32, 32)  # CIFAR-10 size
        output1 = model1(x1)
        print(f"Saída VGG Básica: {output1.shape}")
        assert output1.shape == torch.Size([2, 10]), "Erro no formato de saída da VGG Básica."
        print("VGG Básica testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar VGG Básica: {e}")

    # Teste 2: VGG com configuração personalizada
    print("\n--- Teste 2: VGG Personalizada ---")
    try:
        params_custom = VGGParams(
            num_classes=100,
            min_channels=64,
            max_channels=512,
            dropout_rate=0.3,
            num_layers=6,
            batch_norm=True
        )
        model2 = generate_vgg_architecture(params_custom)
        x2 = torch.randn(2, 3, 32, 32)  # CIFAR-10 size
        output2 = model2(x2)
        print(f"Saída VGG Personalizada: {output2.shape}")
        assert output2.shape == torch.Size([2, 100]), "Erro no formato de saída da VGG Personalizada."
        print("VGG Personalizada testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar VGG Personalizada: {e}")

    # Teste 3: VGG com configuração mínima
    print("\n--- Teste 3: VGG Mínima ---")
    try:
        params_min = VGGParams(
            num_classes=5,
            min_channels=16,
            max_channels=64,
            dropout_rate=0.2,
            num_layers=3,
            batch_norm=False
        )
        model3 = generate_vgg_architecture(params_min)
        x3 = torch.randn(2, 3, 32, 32)  # CIFAR-10 size
        output3 = model3(x3)
        print(f"Saída VGG Mínima: {output3.shape}")
        assert output3.shape == torch.Size([2, 5]), "Erro no formato de saída da VGG Mínima."
        print("VGG Mínima testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar VGG Mínima: {e}")

    print("\n--- Todos os testes concluídos ---")