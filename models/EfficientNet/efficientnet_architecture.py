import math
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union, Type

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.hardtanh(x + 3, 0., 6.) / 6.

class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'swish': Swish,
    'hardswish': HardSwish,
    'mish': Mish,
    'silu': nn.SiLU,
    'leakyrelu': lambda: nn.LeakyReLU(negative_slope=0.1)
}

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_factor: int = 4,
        activation_fn: nn.Module = Swish(),
        gate_fn: nn.Module = nn.Sigmoid()
    ):
        super().__init__()
        squeeze_channels = max(1, in_channels // squeeze_factor)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, 1),
            activation_fn,
            nn.Conv2d(squeeze_channels, in_channels, 1),
            gate_fn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        se_ratio: float = 0.25,
        dropout_rate: float = 0.2,
        activation_fn: nn.Module = Swish(),
        use_batch_norm: bool = True,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 1e-5,
        se_gate_fn: nn.Module = nn.Sigmoid()
    ):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        self.dropout_rate = dropout_rate
        expanded_channels = in_channels * expansion_factor

        norm_layer = lambda c: nn.BatchNorm2d(c, momentum=batch_norm_momentum, eps=batch_norm_epsilon) if use_batch_norm else nn.Identity()

        layers = []
        
        # Expansion phase
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                norm_layer(expanded_channels),
                activation_fn
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels,
                     bias=False),
            norm_layer(expanded_channels),
            activation_fn
        ])

        # Squeeze and Excitation
        if se_ratio > 0:
            layers.append(SqueezeExcitation(
                expanded_channels,
                squeeze_factor=int(1/se_ratio),
                activation_fn=activation_fn,
                gate_fn=se_gate_fn
            ))

        # Projection phase
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            if self.training and self.dropout_rate > 0:
                keep_prob = 1 - self.dropout_rate
                mask = torch.rand([x.shape[0], 1, 1, 1], device=x.device) < keep_prob
                result = self.layers(x)
                scaled_result = result / keep_prob
                output = torch.where(mask, scaled_result, torch.zeros_like(scaled_result))
                return x + output
            else:
                return x + self.layers(x)
        else:
            return self.layers(x)

class EfficientNet(nn.Module):
    def __init__(
        self,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        resolution: int = 224,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        activation_fn: Union[str, nn.Module] = 'swish',
        use_batch_norm: bool = True,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 1e-5,
        se_ratio: float = 0.25,
        stem_channels: int = 32,
        head_channels: int = 1280,
        weight_init_fn: Optional[Callable] = None,
        conv_kernel_initializer: Optional[Callable] = None,
        use_se: bool = True,
        se_gate_fn: nn.Module = nn.Sigmoid()
    ):
        super().__init__()
        
        # Process activation function
        if isinstance(activation_fn, str):
            if activation_fn not in ACTIVATION_FUNCTIONS:
                raise ValueError(f"Activation function must be one of {list(ACTIVATION_FUNCTIONS.keys())}")
            activation_fn = ACTIVATION_FUNCTIONS[activation_fn]()
        
        # Base architecture configuration
        base_config = [
            # (kernel_size, channels, layers, stride, expansion_factor)
            (3, stem_channels, 1, 1, 1),      # Stage 1
            (3, 16, 1, 1, 6),      # Stage 2
            (5, 24, 2, 2, 6),      # Stage 3
            (3, 40, 2, 2, 6),      # Stage 4
            (5, 80, 3, 2, 6),      # Stage 5
            (5, 112, 3, 1, 6),     # Stage 6
            (5, 192, 4, 2, 6),     # Stage 7
            (3, 320, 1, 1, 6),     # Stage 8
        ]

        # Adjust channels and layers based on width and depth multipliers
        config = []
        for k, c, l, s, e in base_config:
            c_out = self._round_filters(c, width_multiplier)
            num_layers = self._round_repeats(l, depth_multiplier)
            config.append((k, c_out, num_layers, s, e))

        # Initial conv layer
        channels = config[0][1]
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=batch_norm_momentum, eps=batch_norm_epsilon) if use_batch_norm else nn.Identity(),
            activation_fn
        )

        # Build MBConv blocks
        layers = []
        in_channels = channels
        total_blocks = sum(layers for _, _, layers, _, _ in config)
        block_idx = 0
        
        for stage_idx, (kernel_size, out_channels, num_layers, stride, expansion_factor) in enumerate(config):
            for layer_idx in range(num_layers):
                # Calculate drop connect rate
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                # Only use stride in the first layer of each stage
                curr_stride = stride if layer_idx == 0 else 1
                layers.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=curr_stride,
                        expansion_factor=expansion_factor,
                        se_ratio=se_ratio if use_se else 0,
                        dropout_rate=drop_rate,
                        activation_fn=activation_fn,
                        use_batch_norm=use_batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        batch_norm_epsilon=batch_norm_epsilon,
                        se_gate_fn=se_gate_fn
                    )
                )
                in_channels = out_channels
                block_idx += 1

        self.blocks = nn.Sequential(*layers)

        # Final stage
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels, momentum=batch_norm_momentum, eps=batch_norm_epsilon) if use_batch_norm else nn.Identity(),
            activation_fn
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_channels, num_classes)
        )

        # Initialize weights
        if conv_kernel_initializer is not None:
            self._init_conv_weights(conv_kernel_initializer)
        if weight_init_fn is not None:
            self.apply(weight_init_fn)
        else:
            self._initialize_weights()

    def _round_filters(self, filters: int, width_multiplier: float) -> int:
        """Round number of filters based on width multiplier."""
        multiplier = width_multiplier
        divisor = 8
        filters *= multiplier
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats: int, depth_multiplier: float) -> int:
        """Round number of layers based on depth multiplier."""
        return int(math.ceil(depth_multiplier * repeats))

    def _init_conv_weights(self, initializer: Callable):
        """Initialize convolutional layers with custom initializer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_weights(self):
        """Initialize model weights with default initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.classifier(x)
        return x

def generate_efficientnet_architecture(
    model_variant: str = 'B0',
    num_classes: int = 1000,
    dropout_rate: float = 0.2,
    activation_fn: Union[str, nn.Module] = 'swish',
    use_batch_norm: bool = True,
    batch_norm_momentum: float = 0.1,
    batch_norm_epsilon: float = 1e-5,
    se_ratio: float = 0.25,
    stem_channels: Optional[int] = None,
    head_channels: Optional[int] = None,
    weight_init_fn: Optional[Callable] = None,
    conv_kernel_initializer: Optional[Callable] = None,
    use_se: bool = True,
    drop_connect_rate: float = 0.2
) -> EfficientNet:
    """
    Generate an EfficientNet model based on the specified variant.
    
    Args:
        model_variant (str): Which EfficientNet model to create ('B0' through 'B7')
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        activation_fn (str or nn.Module): Activation function to use
        use_batch_norm (bool): Whether to use batch normalization
        batch_norm_momentum (float): Momentum for batch normalization
        batch_norm_epsilon (float): Epsilon for batch normalization
        se_ratio (float): Squeeze-and-excitation ratio
        stem_channels (int, optional): Number of channels in the stem
        head_channels (int, optional): Number of channels in the head
        weight_init_fn (callable, optional): Custom weight initialization function
        conv_kernel_initializer (callable, optional): Custom conv kernel initializer
        use_se (bool): Whether to use squeeze-and-excitation
        drop_connect_rate (float): Drop connect rate
    
    Returns:
        EfficientNet: Configured model instance
    """
    # Model scaling coefficients for different variants
    # (width_multiplier, depth_multiplier, resolution, dropout_rate)
    variants = {
        'B0': (1.0, 1.0, 224, 0.2),
        'B1': (1.0, 1.1, 240, 0.2),
        'B2': (1.1, 1.2, 260, 0.3),
        'B3': (1.2, 1.4, 300, 0.3),
        'B4': (1.4, 1.8, 380, 0.4),
        'B5': (1.6, 2.2, 456, 0.4),
        'B6': (1.8, 2.6, 528, 0.5),
        'B7': (2.0, 3.1, 600, 0.5),
    }
    
    if model_variant not in variants:
        raise ValueError(f"Model variant must be one of {list(variants.keys())}")
    
    width_mult, depth_mult, resolution, base_dropout = variants[model_variant]
    
    # Use provided dropout_rate if specified, otherwise use the base dropout rate
    final_dropout = dropout_rate if dropout_rate is not None else base_dropout
    
    # Set default stem and head channels if not provided
    if stem_channels is None:
        stem_channels = 32
    if head_channels is None:
        head_channels = 1280
    
    return EfficientNet(
        width_multiplier=width_mult,
        depth_multiplier=depth_mult,
        resolution=resolution,
        num_classes=num_classes,
        dropout_rate=final_dropout,
        drop_connect_rate=drop_connect_rate,
        activation_fn=activation_fn,
        use_batch_norm=use_batch_norm,
        batch_norm_momentum=batch_norm_momentum,
        batch_norm_epsilon=batch_norm_epsilon,
        se_ratio=se_ratio,
        stem_channels=stem_channels,
        head_channels=head_channels,
        weight_init_fn=weight_init_fn,
        conv_kernel_initializer=conv_kernel_initializer,
        use_se=use_se
    )

if __name__ == "__main__":
    # Example usage with extended parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a customized EfficientNet-B0 model
    model = generate_efficientnet_architecture(
        model_variant='B0',
        num_classes=1000,
        activation_fn='mish',  # Using Mish activation
        use_batch_norm=True,
        batch_norm_momentum=0.1,
        se_ratio=0.25,
        stem_channels=32,
        head_channels=1280,
        use_se=True,
        drop_connect_rate=0.2
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(input_tensor)
    
    print(f"Model Output Shape: {output.shape}")  # Should be [4, 1000]
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
