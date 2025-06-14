import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, batch_norm=True):
        super(_DenseLayer, self).__init__()
        self.batch_norm = batch_norm
        
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=1, bias=False)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.batch_norm:
            out = self.bn1(x)
        else:
            out = x
        out = F.relu(out)
        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, batch_norm=True):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                drop_rate,
                batch_norm
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features, batch_norm=True):
        super(_Transition, self).__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_classes=1000, drop_rate=0, batch_norm=True,
                 weight_init_fn=None):
        super(DenseNet, self).__init__()
        
        # Primeira camada convolucional
        num_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Blocos densos e camadas de transição
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                batch_norm=batch_norm
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    batch_norm=batch_norm
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # Camada final de batch norm
        if batch_norm:
            self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Classificador
        self.classifier = nn.Linear(num_features, num_classes)

        # Inicialização dos pesos
        if weight_init_fn is not None:
            self.apply(weight_init_fn)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def generate_densenet_architecture(
    min_growth_rate=12,
    max_growth_rate=48,
    min_blocks=3,
    max_blocks=5,
    min_layers_per_block=4,
    max_layers_per_block=16,
    num_classes=1000,
    drop_rate=0,
    batch_norm=True
):
    """
    Gera uma arquitetura DenseNet com parâmetros aleatórios.
    
    Args:
        min_growth_rate (int): Taxa de crescimento mínima
        max_growth_rate (int): Taxa de crescimento máxima
        min_blocks (int): Número mínimo de blocos densos
        max_blocks (int): Número máximo de blocos densos
        min_layers_per_block (int): Número mínimo de camadas por bloco
        max_layers_per_block (int): Número máximo de camadas por bloco
        num_classes (int): Número de classes de saída
        drop_rate (float): Taxa de dropout
        batch_norm (bool): Se deve usar batch normalization
    
    Returns:
        DenseNet: Uma instância da rede DenseNet com arquitetura aleatória
    """
    import random
    
    # Gera parâmetros aleatórios
    growth_rate = random.randint(min_growth_rate, max_growth_rate)
    num_blocks = random.randint(min_blocks, max_blocks)
    block_config = [random.randint(min_layers_per_block, max_layers_per_block) 
                   for _ in range(num_blocks)]
    
    # Cria e retorna a rede
    return DenseNet(
        growth_rate=growth_rate,
        block_config=tuple(block_config),
        num_classes=num_classes,
        drop_rate=drop_rate,
        batch_norm=batch_norm
    ) 