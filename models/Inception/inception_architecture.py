import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            conv_block(in_channels, 48, kernel_size=1),
            conv_block(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3dbl = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3, padding=1),
            conv_block(96, 96, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3, padding=1),
            conv_block(96, 96, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7 = nn.Sequential(
            conv_block(in_channels, c7, kernel_size=1),
            conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7dbl = nn.Sequential(
            conv_block(in_channels, c7, kernel_size=1),
            conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7dbl = self.branch7x7dbl(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7x3 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(192, 192, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7x3 = self.branch7x7x3(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        # Branch 2: 1x1 conv -> (1x3 conv, 3x1 conv) parallel
        self.branch3x3_1x1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_1x3 = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_3x1 = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 3: 1x1 conv -> 3x3 conv -> (1x3 conv, 3x1 conv) parallel
        self.branch3x3dbl_1x1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_3x3 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_1x3 = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3x1 = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        # Branch 2
        branch3x3_pre = self.branch3x3_1x1(x)
        branch3x3 = torch.cat([
            self.branch3x3_1x3(branch3x3_pre),
            self.branch3x3_3x1(branch3x3_pre)
        ], 1)

        # Branch 3
        branch3x3dbl_pre = self.branch3x3dbl_1x1(x)
        branch3x3dbl_pre = self.branch3x3dbl_3x3(branch3x3dbl_pre)
        branch3x3dbl = torch.cat([
            self.branch3x3dbl_1x3(branch3x3dbl_pre),
            self.branch3x3dbl_3x1(branch3x3dbl_pre)
        ], 1)

        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        # After adaptive pooling to 1x1, the next conv should also be 1x1
        self.conv1 = conv_block(128, 768, kernel_size=1, batch_norm=False)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Pool to 1x1 before conv1
        # N x 128 x 1 x 1
        x = self.conv1(x)
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x num_classes
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=10, aux_logits=False, transform_input=False, 
                 dropout_rate=0.5, init_weights=True, weight_init_fn=None):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.init_weights = init_weights
        self.weight_init_fn = weight_init_fn

        # Reduzindo o stride inicial para não diminuir tanto a dimensão
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        # Removendo primeiro maxpool para preservar dimensões
        
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduzindo kernel size

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)
        
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        
        # Removendo camadas extras para evitar redução excessiva

        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, num_classes)

        if init_weights:
            self._initialize_weights(weight_init_fn)

    def _initialize_weights(self, weight_init_fn):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if weight_init_fn is not None:
                    weight_init_fn(m)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # Stem
        x = self.Conv2d_1a_3x3(x)  # 32x32
        x = self.Conv2d_2a_3x3(x)  # 32x32
        x = self.Conv2d_2b_3x3(x)  # 32x32
        x = self.Conv2d_3b_1x1(x)  # 32x32
        x = self.Conv2d_4a_3x3(x)  # 32x32
        x = self.maxpool2(x)       # 16x16

        # Inception blocks
        x = self.Mixed_5b(x)       # 16x16
        x = self.Mixed_5c(x)       # 16x16
        x = self.Mixed_5d(x)       # 16x16
        x = self.Mixed_6a(x)       # 8x8
        x = self.Mixed_6b(x)       # 8x8

        aux = None
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux
        return x

def generate_inception_architecture(
    num_classes=10,
    aux_logits=True,
    transform_input=False,
    dropout_rate=0.5,
    init_weights=True,
    weight_init_fn=None
):
    """
    Gera uma instância da InceptionV3 baseada em parâmetros que permitem a variação de hiperparâmetros.

    A arquitetura InceptionV3 possui uma estrutura fixa de blocos para manter suas características de desempenho.
    A "modularidade" aqui se refere à capacidade de configurar hiperparâmetros e a inclusão/exclusão de
    camadas auxiliares e inicialização de pesos, em vez de alterar a topologia fundamental da rede.

    Args:
        num_classes (int): Número de classes de saída.
        aux_logits (bool): Se True, inclui camadas auxiliares para treinamento de gradientes profundos.
        transform_input (bool): Se True, pré-processa a entrada para o formato esperado pelo InceptionV3.
        dropout_rate (float): Taxa de dropout a ser aplicada na camada final.
        init_weights (bool): Se True, inicializa os pesos da rede.
        weight_init_fn (callable, optional): Função para inicializar os pesos. Se None, usa kaiming_normal_.

    Returns:
        InceptionV3: Uma instância do modelo InceptionV3.
    """

    model = InceptionV3(
        num_classes=num_classes,
        aux_logits=aux_logits,
        transform_input=transform_input,
        dropout_rate=dropout_rate,
        init_weights=init_weights,
        weight_init_fn=weight_init_fn
    )

    return model

if __name__ == '__main__':
    print("\n--- Exemplo de Uso do InceptionV3 ---")

    # Exemplo 1: Modelo InceptionV3 básico
    print("\nCriando um modelo InceptionV3 básico (sem logits auxiliares, 1000 classes):")
    model_basic = generate_inception_architecture(num_classes=10, aux_logits=False)
    print(model_basic)
    print(f"Número total de parâmetros: {sum(p.numel() for p in model_basic.parameters() if p.requires_grad)}")
    """
    # Testando o forward pass
    input_tensor = torch.randn(1, 3, 299, 299) # InceptionV3 espera 299x299
    output, aux_output = model_basic(input_tensor)
    print(f"Saída do modelo (main): {output.shape}")
    print(f"Saída do modelo (aux): {aux_output}") # Deve ser None

    # Exemplo 2: Modelo InceptionV3 com logits auxiliares
    print("\nCriando um modelo InceptionV3 com logits auxiliares (10 classes):")
    model_aux = generate_inception_architecture(num_classes=10, aux_logits=True)
    print(model_aux)

    # Testando o forward pass com logits auxiliares
    model_aux.train() # Colocar em modo de treinamento para ativar logits auxiliares
    output, aux_output = model_aux(input_tensor)
    print(f"Saída do modelo (main): {output.shape}")
    print(f"Saída do modelo (aux): {aux_output.shape}")

    # Exemplo 3: Modelo InceptionV3 com dropout_rate diferente e inicialização de pesos personalizada
    print("\nCriando um modelo InceptionV3 com dropout_rate=0.7 e inicialização Xavier uniform:")
    def xavier_uniform_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    model_custom_init = generate_inception_architecture(
        num_classes=50,
        dropout_rate=0.7,
        init_weights=True,
        weight_init_fn=xavier_uniform_init
    )
    print(model_custom_init)
    output, aux_output = model_custom_init(input_tensor)
    print(f"Saída do modelo (main): {output.shape}")
    """


