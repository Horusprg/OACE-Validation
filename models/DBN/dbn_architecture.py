import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable

# --- 1. Classe para uma Restricted Boltzmann Machine (RBM) ---
# Uma DBN é uma pilha de RBMs. Esta classe define uma única RBM.
# Para simplificar a arquitetura da DBN, esta RBM será apenas um módulo linear.
# O treinamento real de uma RBM é mais complexo (amostragem de Gibbs, contraste divergente)
# e não está incluído aqui, pois o foco é a estrutura da DBN.
class RBMLayer(nn.Module):
    """
    Camada de Restricted Boltzmann Machine (RBM).
    
    Esta classe implementa uma camada RBM simplificada, incluindo:
    - Transformação linear
    - Função de ativação
    - Dropout (opcional)
    - Bias para unidades visíveis e ocultas
    
    A inicialização dos pesos é otimizada para melhor convergência.
    """
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        activation_fn: nn.Module = nn.Sigmoid,
        dropout_rate: float = 0.0
    ):
        super(RBMLayer, self).__init__()
        
        # Camada linear com inicialização otimizada
        self.linear_layer = nn.Linear(n_visible, n_hidden)
        self._initialize_weights(activation_fn)
        
        self.activation_fn = activation_fn()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Bias para as unidades visíveis e ocultas
        self.register_buffer('v_bias', torch.zeros(n_visible))
        self.register_buffer('h_bias', torch.zeros(n_hidden))
        
    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos da camada RBM de forma otimizada.
        
        Args:
            activation_fn: Função de ativação usada na camada
        """
        if isinstance(activation_fn, nn.Sigmoid):
            # Para RBMs com unidades binárias
            nn.init.xavier_normal_(self.linear_layer.weight, gain=1.0)
        elif isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU)):
            # Para RBMs com unidades ReLU
            nn.init.kaiming_normal_(self.linear_layer.weight, nonlinearity='relu')
        else:
            # Fallback para outras funções de ativação
            nn.init.xavier_normal_(self.linear_layer.weight)
            
        # Inicializa o bias com zeros
        nn.init.zeros_(self.linear_layer.bias)
        
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada RBM.
        
        Args:
            v: Tensor de entrada (unidades visíveis)
            
        Returns:
            Tensor de saída (probabilidades das unidades ocultas)
        """
        h_prob = self.activation_fn(self.linear_layer(v))
        if self.dropout is not None:
            h_prob = self.dropout(h_prob)
        return h_prob

# --- 2. Classe para a Deep Belief Network (DBN) ---
class DBNClassifier(nn.Module):
    """
    Classificador MLP para a Deep Belief Network.
    
    Esta classe implementa o classificador final da DBN, incluindo:
    - Múltiplas camadas ocultas
    - Função de ativação
    - Dropout (opcional)
    - Inicialização otimizada dos pesos
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        super(DBNClassifier, self).__init__()
        
        layers = []
        current_dim = input_dim

        # Constrói as camadas ocultas
        for h_dim in hidden_layers:
            linear_layer = nn.Linear(current_dim, h_dim)
            self._initialize_weights(linear_layer, activation_fn)
            layers.append(linear_layer)
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Camada de saída
        output_layer = nn.Linear(current_dim, output_dim)
        self._initialize_weights(output_layer, activation_fn)
        layers.append(output_layer)
        
        self.classifier = nn.Sequential(*layers)
        
    def _initialize_weights(self, layer: nn.Linear, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos de uma camada linear de forma otimizada.
        
        Args:
            layer: Camada linear a ser inicializada
            activation_fn: Função de ativação usada na camada
        """
        if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif isinstance(activation_fn, (nn.Sigmoid, nn.Tanh)):
            nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.xavier_normal_(layer.weight)
            
        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do classificador.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Logits da rede
        """
        return self.classifier(x)

class DBN(nn.Module):
    """
    Implementação de uma Deep Belief Network (DBN).
    
    Esta rede neural consiste em:
    1. Uma pilha de Restricted Boltzmann Machines (RBMs)
    2. Um classificador MLP no topo
    
    A rede é otimizada para:
    - Inicialização eficiente dos pesos
    - Treinamento estável
    - Regularização flexível
    """
    def __init__(
        self,
        input_dim: int,
        rbm_hidden_dims: List[int],
        output_dim: int,
        rbm_activation_fn: nn.Module = nn.Sigmoid,
        classifier_activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        classifier_hidden_layers: List[int] = None
    ):
        super(DBN, self).__init__()

        if classifier_hidden_layers is None:
            classifier_hidden_layers = []

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate deve estar entre 0.0 e 1.0")

        self.input_dim = input_dim
        self.rbm_hidden_dims = rbm_hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.classifier_hidden_layers = classifier_hidden_layers

        # Constrói a pilha de RBMs
        rbm_layers = []
        current_rbm_input_dim = input_dim
        
        for i, h_dim in enumerate(rbm_hidden_dims):
            rbm_layer = RBMLayer(
                n_visible=current_rbm_input_dim,
                n_hidden=h_dim,
                activation_fn=rbm_activation_fn,
                dropout_rate=dropout_rate if i < len(rbm_hidden_dims) - 1 else 0.0
            )
            rbm_layers.append(rbm_layer)
            current_rbm_input_dim = h_dim

        self.rbm_stack = nn.Sequential(*rbm_layers)

        # Constrói o classificador
        self.classifier = DBNClassifier(
            input_dim=current_rbm_input_dim,
            hidden_layers=classifier_hidden_layers,
            output_dim=output_dim,
            activation_fn=classifier_activation_fn,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da rede.
        
        Args:
            x: Tensor de entrada (pode ser 2D ou 3D)
            
        Returns:
            Logits da rede
        """
        x = x.view(x.size(0), -1)
        features = self.rbm_stack(x)
        return self.classifier(features)

# --- 3. Função para Gerar a Arquitetura DBN ---
def generate_dbn_architecture(
    input_dim: int,
    output_dim: int,
    num_rbm_layers: int,
    min_rbm_neurons: int,
    max_rbm_neurons: int,
    num_classifier_hidden_layers: int,
    min_classifier_neurons: int,
    max_classifier_neurons: int,
    rbm_activation_function_choice: str = 'sigmoid',
    classifier_activation_function_choice: str = 'relu',
    dropout_rate: float = 0.0
) -> DBN:
    """
    Gera uma arquitetura DBN otimizada.
    
    Esta função cria uma DBN com:
    - Pilha de RBMs com dimensões aleatórias
    - Classificador MLP configurável
    - Funções de ativação específicas para cada parte
    - Regularização configurável
    
    Args:
        input_dim: Dimensão da camada de entrada
        output_dim: Dimensão da camada de saída
        num_rbm_layers: Número de camadas RBM
        min_rbm_neurons: Mínimo de unidades ocultas por RBM
        max_rbm_neurons: Máximo de unidades ocultas por RBM
        num_classifier_hidden_layers: Número de camadas ocultas no classificador
        min_classifier_neurons: Mínimo de neurônios por camada do classificador
        max_classifier_neurons: Máximo de neurônios por camada do classificador
        rbm_activation_function_choice: Ativação para RBMs ('sigmoid', 'relu')
        classifier_activation_function_choice: Ativação para classificador ('relu', 'leaky_relu', 'elu', 'selu', 'gelu')
        dropout_rate: Taxa de dropout (0-1)
        
    Returns:
        DBN: Rede neural configurada
        
    Raises:
        ValueError: Se os parâmetros forem inválidos
    """
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError("dropout_rate deve estar entre 0.0 e 1.0")
    if min_rbm_neurons > max_rbm_neurons:
        raise ValueError("min_rbm_neurons deve ser menor que max_rbm_neurons")
    if min_classifier_neurons > max_classifier_neurons:
        raise ValueError("min_classifier_neurons deve ser menor que max_classifier_neurons")

    # Mapeamento de funções de ativação
    activation_map = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    
    rbm_activation_fn = activation_map.get(rbm_activation_function_choice.lower())
    if rbm_activation_fn is None:
        raise ValueError(f"Função de ativação RBM '{rbm_activation_function_choice}' não suportada.")

    classifier_activation_fn = activation_map.get(classifier_activation_function_choice.lower())
    if classifier_activation_fn is None:
        raise ValueError(f"Função de ativação do classificador '{classifier_activation_function_choice}' não suportada.")

    # Gera as dimensões para as camadas ocultas das RBMs
    rbm_hidden_dims = [
        torch.randint(low=min_rbm_neurons, high=max_rbm_neurons + 1, size=(1,)).item()
        for _ in range(num_rbm_layers)
    ]

    # Gera as dimensões para as camadas ocultas do classificador
    classifier_hidden_dims = [
        torch.randint(low=min_classifier_neurons, high=max_classifier_neurons + 1, size=(1,)).item()
        for _ in range(num_classifier_hidden_layers)
    ]

    print(f"Arquitetura DBN gerada:")
    print(f"- Pilha de RBMs (unidades ocultas): {rbm_hidden_dims}")
    print(f"- Ativação RBM: {rbm_activation_function_choice}")
    print(f"- Classificador (camadas ocultas): {classifier_hidden_dims}")
    print(f"- Ativação Classificador: {classifier_activation_function_choice}")
    print(f"- Dropout: {dropout_rate}")

    return DBN(
        input_dim=input_dim,
        rbm_hidden_dims=rbm_hidden_dims,
        output_dim=output_dim,
        rbm_activation_fn=rbm_activation_fn,
        classifier_activation_fn=classifier_activation_fn,
        dropout_rate=dropout_rate,
        classifier_hidden_layers=classifier_hidden_dims
    )