import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time
import psutil
import os
from typing import Dict, Tuple, Any
from thop import profile

def get_model_from_config(architecture_config: Dict[str, Any]) -> nn.Module:
    """
    Instancia um modelo PyTorch baseado na configuração fornecida.
    
    Args:
        architecture_config: Dicionário com a configuração da arquitetura
        
    Returns:
        Modelo PyTorch instanciado
    """
    model_type = architecture_config.get("model_type")
    if model_type is None:
        raise ValueError("O campo 'model_type' é obrigatório em architecture_config.")
    
    model_type = architecture_config.get('model_type', 'ResNet')
    
    model_kwargs = {k: v for k, v in architecture_config.items() if k != "model_type"}
    
    if model_type == 'CNN':
        from models.CNN.cnn_architectue import generate_cnn_architecture
        
        default_kwargs = {
            "in_channels": 3,
            "num_classes": 10,
            "num_conv_blocks": 3,
            "min_channels_per_block": 32,
            "max_channels_per_block": 256,
            "kernel_size_choice": "random",
            "stride_choice": 1,
            "padding_choice": "same",
            "pooling_type_choice": "max",
            "pooling_kernel_size": 2,
            "pooling_stride": 2,
            "activation_function_choice": "relu",
            "dropout_rate": 0.0,
            "batch_norm": True,
            "num_fc_layers": 1,
            "min_fc_neurons": 64,
            "max_fc_neurons": 512
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_cnn_architecture(**model_kwargs)
    
    elif model_type == 'VGG':
        from models.VGG.vgg_architecture import generate_vgg_architecture
        
        default_kwargs = {
            "num_classes": 10,
            "in_channels": 3,
            "dropout_rate": 0.5,
            "batch_norm": False,
            "weight_init_fn": "kaiming",
            "min_conv_layers_per_block": 1,
            "max_conv_layers_per_block": 3,
            "min_channels": 64,
            "max_channels": 512,
            "num_conv_blocks": 5,
            "min_fc_neurons": 1024,
            "max_fc_neurons": 4096,
            "num_fc_layers": 2
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_vgg_architecture(**model_kwargs)
    
    elif model_type == 'InceptionV3':
        from models.Inception.inception_architecture import generate_inception_architecture
        
        default_kwargs = {
            "num_classes": 10,
            "aux_logits": False,
            "transform_input": False,
            "dropout_rate": 0.5,
            "init_weights": True,
            "weight_init_fn": None
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_inception_architecture(**model_kwargs)
        
    elif model_type == 'ResNet':
        from models.ResNet.resnet_architecture import generate_resnet_architecture
        
        default_kwargs = {
            "in_channels": 3,
            "num_classes": 10,
            "num_stages": 4,
            "min_blocks_per_stage": 1,
            "max_blocks_per_stage": 4,
            "min_base_channels": 16,
            "max_base_channels": 64,
            "block_type_choice": "basic",
            "activation_function_choice": "relu",
            "dropout_rate": 0.0,
            "batch_norm": True
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_resnet_architecture(**model_kwargs)
        
    elif model_type == 'MLP':
        from models.MLP.mlp_architecture import generate_mlp_architecture
        
        default_kwargs = {
            "input_dim": 3072,
            "output_dim": 10,
            "num_hidden_layers": 2,
            "min_neurons_per_layer": 64,
            "max_neurons_per_layer": 512,
            "activation_function_choice": "relu",
            "dropout_rate": 0.0,
            "batch_norm": False
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_mlp_architecture(**model_kwargs)
    
    elif model_type == 'DBN':
        from models.DBN.dbn_architecture import generate_dbn_architecture
        
        default_kwargs = {
            "input_dim": 3072,
            "output_dim": 10,
            "num_rbm_layers": 2,
            "min_rbm_neurons": 64,
            "max_rbm_neurons": 512,
            "num_classifier_hidden_layers": 2,
            "min_classifier_neurons": 64,
            "max_classifier_neurons": 512,
            "rbm_activation_function_choice": "sigmoid",
            "classifier_activation_function_choice": "relu",
            "dropout_rate": 0.0
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_dbn_architecture(**model_kwargs)
        
    elif model_type == 'EfficientNet':
        from models.EfficientNet.efficientnet_architecture import generate_efficientnet_architecture
        
        default_kwargs = {
            "model_variant": "B0",
            "num_classes": 1000,
            "dropout_rate": 0.2,
            "activation_fn": "swish",
            "use_batch_norm": True,
            "batch_norm_momentum": 0.1,
            "batch_norm_epsilon": 1e-5,
            "se_ratio": 0.25,
            "stem_channels": None,
            "head_channels": None,
            "weight_init_fn": None,
            "conv_kernel_initializer": None,
            "use_se": True,
            "drop_connect_rate": 0.2
        }
        
        model_kwargs = {**default_kwargs, **model_kwargs}
        return generate_efficientnet_architecture(**model_kwargs)
        
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")

def collect_cost_metrics(model: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Coleta métricas de custo do modelo.
    
    Args:
        model: Modelo PyTorch
        device: Dispositivo de execução
        
    Returns:
        Dicionário com métricas de custo (total_params, avg_inference_time, memory_used_mb, gflops)
    """
    # Total de parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    
    # Tempo de inferência
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Adapta entrada para MLP/DBN
    if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
        input_tensor = input_tensor.view(batch_size, -1)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Medição do tempo
    inference_times = []
    for _ in range(100):
        start_time = time.time()
        _ = model(input_tensor)
        inference_times.append(time.time() - start_time)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    # Memória usada
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated(device) / 1024**2  # Em MB
    else:
        # Estimativa para CPU: memória dos parâmetros e buffers
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        # Aproximação para ativações (assume batch_size=128, imagem 32x32)
        activation_memory = (128 * 3 * 32 * 32 * 4) / 1024**2  # Entrada em MB
        memory_used = param_memory + buffer_memory + activation_memory
    
    # FLOPs usando thop
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
        sample_input = sample_input.view(1, -1)
    flops, _ = profile(model, inputs=(sample_input,))
    gflops = flops / 1e9
    
    return {
        'total_params': float(total_params),
        'avg_inference_time': float(avg_inference_time),
        'memory_used_mb': float(memory_used),
        'gflops': float(gflops)
    }

def train_and_evaluate_for_oace(
    architecture_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 5
) -> Dict[str, float]:
    """
    Treina e avalia um modelo para cálculo do score OACE.
    
    Args:
        architecture_config: Configuração da arquitetura
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        device: Dispositivo de execução
        epochs: Número de épocas para treinamento
        
    Returns:
        Dicionário com todas as métricas coletadas
    """
    # Instanciando modelo
    model = get_model_from_config(architecture_config)
    model = model.to(device)
    
    # Configurando treinamento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Treinamento
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Lidando com saída do InceptionV3 que pode retornar tupla
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Pega apenas a saída principal
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Avaliação
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Lidando com saída do InceptionV3 que pode retornar tupla
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Pega apenas a saída principal
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas de assertividade
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Métricas de custo
    cost_metrics = collect_cost_metrics(model, device)
    
    # Combinando todas as métricas
    metrics = {
        'model': architecture_config.get('model_type'),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'total_params': cost_metrics['total_params'],
        'avg_inference_time': cost_metrics['avg_inference_time'],
        'memory_used_mb': cost_metrics['memory_used_mb'],
        'gflops': cost_metrics['gflops']
    }
    
    return metrics