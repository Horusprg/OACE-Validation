import torch
import torch.nn as nn
from typing import List, Optional, Callable

class MLPLayer(nn.Module):
    """
    Camada individual de uma Multi-Layer Perceptron.
    
    Esta classe implementa uma camada completa da MLP, incluindo:
    - Camada linear (transformação)
    - Normalização em lote (opcional)
    - Função de ativação
    - Dropout (opcional)
    
    A inicialização dos pesos é otimizada para melhor convergência.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = False
    ):
        super(MLPLayer, self).__init__()
        
        # Camada linear com inicialização otimizada
        self.linear = nn.Linear(in_features, out_features)
        self._initialize_weights(activation_fn)
            
        # Componentes opcionais
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos da camada linear de forma otimizada.
        
        Args:
            activation_fn: Função de ativação usada na camada
        """
        if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU)):
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        elif isinstance(activation_fn, (nn.Sigmoid, nn.Tanh)):
            nn.init.xavier_normal_(self.linear.weight)
        elif isinstance(activation_fn, (nn.ELU, nn.SELU)):
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.linear.weight)
            
        # Inicializa o bias com zeros
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor processado pela camada
        """
        x = self.linear(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class MLPOutputLayer(nn.Module):
    """
    Camada de saída da MLP.
    
    Esta camada implementa a transformação final da rede,
    mapeando as features para o espaço de classes.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        super(MLPOutputLayer, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self._initialize_weights()
            
    def _initialize_weights(self) -> None:
        """
        Inicializa os pesos da camada de saída.
        Usa inicialização Xavier para melhor estabilidade.
        """
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada de saída.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Logits da rede
        """
        return self.linear(x)

class MLP(nn.Module):
    """
    Implementação de uma Multi-Layer Perceptron (MLP).
    
    Esta rede neural consiste em:
    1. Uma sequência de camadas ocultas (MLPLayer)
    2. Uma camada de saída (MLPOutputLayer)
    
    Cada camada oculta pode incluir:
    - Transformação linear
    - Normalização em lote (opcional)
    - Função de ativação
    - Dropout (opcional)
    
    A rede é otimizada para:
    - Inicialização eficiente dos pesos
    - Treinamento estável
    - Regularização flexível
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = False
    ):
        super(MLP, self).__init__()

        if not isinstance(hidden_layers, list) or not all(isinstance(x, int) for x in hidden_layers):
            raise ValueError("hidden_layers deve ser uma lista de inteiros")
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1")

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        layers = []
        current_dim = input_dim

        # Constrói as camadas ocultas
        for h_dim in hidden_layers:
            layers.append(
                MLPLayer(
                    in_features=current_dim,
                    out_features=h_dim,
                    activation_fn=activation_fn,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm
                )
            )
            current_dim = h_dim

        # Adiciona a camada de saída
        layers.append(
            MLPOutputLayer(
                in_features=current_dim,
                out_features=output_dim
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da rede.
        
        Args:
            x: Tensor de entrada (pode ser 2D ou 3D)
            
        Returns:
            Logits da rede
        """
        x = x.view(x.size(0), -1)
        return self.network(x)

def generate_mlp_architecture(
    input_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    min_neurons_per_layer: int,
    max_neurons_per_layer: int,
    activation_function_choice: str = 'relu',
    dropout_rate: float = 0.0,
    batch_norm: bool = False
) -> MLP:
    """
    Gera uma arquitetura MLP otimizada.
    
    Esta função cria uma MLP com:
    - Número especificado de camadas ocultas
    - Dimensões aleatórias dentro dos limites especificados
    - Função de ativação escolhida
    - Regularização configurável (dropout e batch norm)
    
    Args:
        input_dim: Dimensão da camada de entrada
        output_dim: Dimensão da camada de saída
        num_hidden_layers: Número de camadas ocultas
        min_neurons_per_layer: Mínimo de neurônios por camada
        max_neurons_per_layer: Máximo de neurônios por camada
        activation_function_choice: Função de ativação ('relu', 'leaky_relu', 'elu', 'selu', 'gelu')
        dropout_rate: Taxa de dropout (0-1)
        batch_norm: Se True, aplica normalização em lote
        
    Returns:
        MLP: Rede neural configurada
        
    Raises:
        ValueError: Se os parâmetros forem inválidos
    """
    if not (0 <= dropout_rate <= 1):
        raise ValueError("dropout_rate deve estar entre 0 e 1")
    
    if min_neurons_per_layer > max_neurons_per_layer:
        raise ValueError("min_neurons_per_layer deve ser menor que max_neurons_per_layer")

    # Mapeamento de funções de ativação
    activation_map = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU
    }
    
    activation_fn = activation_map.get(activation_function_choice.lower())
    if activation_fn is None:
        raise ValueError(f"Função de ativação '{activation_function_choice}' não suportada")

    # Gera as dimensões das camadas ocultas
    hidden_layers_dims = [
        torch.randint(low=min_neurons_per_layer, high=max_neurons_per_layer + 1, size=(1,)).item()
        for _ in range(num_hidden_layers)
    ]

    print(f"Arquitetura MLP gerada:")
    print(f"- Camadas ocultas: {hidden_layers_dims}")
    print(f"- Função de ativação: {activation_function_choice}")
    print(f"- Dropout: {dropout_rate}")
    print(f"- Batch Normalization: {batch_norm}")

    return MLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers_dims,
        output_dim=output_dim,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm
    )