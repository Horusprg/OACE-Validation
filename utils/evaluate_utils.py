import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import time
from torchinfo import summary
from thop import profile
from typing import List, Dict

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device
                   ) -> Dict[str, float]:
    """
    Avalia um modelo no CIFAR-10 de forma genérica para warm-up.
    
    Args:
        model: Modelo PyTorch (herda de nn.Module).
        test_loader: DataLoader para dados de teste.
        criterion: Função de perda (ex.: CrossEntropyLoss).
        device: Dispositivo (CPU ou GPU).

    Returns:
        Dict[str, float]: Dicionário com test_loss e test_acc.
    """
    model.to(device)
    model.eval()
    
    test_loss, top1_correct, top5_correct, test_total = 0.0, 0, 0, 0
    all_labels, all_preds = [], []
    
    # Medição de tempo de inferência
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Adapta entrada para MLP/DBN
            if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
                inputs = inputs.view(inputs.size(0), -1)
            
            # Mede tempo de inferência
            start_time = time.time()
            outputs = model(inputs)
            inference_times.append(time.time() - start_time)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # Top-1 e Top-5
            _, predicted_top1 = outputs.max(1)
            top1_correct += predicted_top1.eq(labels).sum().item()
            
            # Top-5
            _, predicted_top5 = outputs.topk(5, dim=1)
            top5_correct += labels.view(-1, 1).eq(predicted_top5).any(dim=1).sum().item()
            
            test_total += labels.size(0)
            
            # Coleta previsões e rótulos para métricas sklearn
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_top1.cpu().numpy())
    
    # Calcula métricas de assertividade
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Calcula métricas de custo
    total_params = sum(p.numel() for p in model.parameters())
    avg_inference_time = np.mean(inference_times)
    
    # Memória usada
    """
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated(device) / 1024**2  # Em MB
    else:
        # Estimativa para CPU: memória dos parâmetros e buffers (aproximação)
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        # Aproximação para ativações (assume batch_size=128, imagem 32x32)
        activation_memory = (128 * 3 * 32 * 32 * 4) / 1024**2  # Entrada em MB
        memory_used = param_memory + buffer_memory + activation_memory
    """
        
    memory_used = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)
    
    # FLOPs - Usa batch_size=2 para evitar problemas com BatchNorm
    sample_input = torch.randn(2, 3, 32, 32).to(device)
    if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
        sample_input = sample_input.view(2, -1)
    
    # Garante que o modelo está em modo de avaliação para o profiling
    model.eval()
    with torch.no_grad():
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
    # Converte para GFLOPs e divide por 2 para obter FLOPs por amostra
    gflops = (flops / 1e9) / 2
    
    test_metrics = {
        # Métricas de assertividade
        'loss': test_loss / test_total,
        'top1_acc': 100.0 * top1_correct / test_total,
        'top5_acc': 100.0 * top5_correct / test_total,
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        # Métricas de custo
        'total_params': float(total_params),
        'avg_inference_time': avg_inference_time,
        'memory_used_mb': float(memory_used),
        'gflops': float(gflops)
    }
    
    print(f"Teste: Loss: {test_metrics['loss']:.4f}, "
          f"Top-1 Acc: {test_metrics['top1_acc']:.2f}%, "
          f"Top-5 Acc: {test_metrics['top5_acc']:.2f}%, "
          f"Precision (macro): {test_metrics['precision_macro']:.4f}, "
          f"Recall (macro): {test_metrics['recall_macro']:.4f}, "
          f"F1 (macro): {test_metrics['f1_macro']:.4f}, "
          f"Params: {test_metrics['total_params']:.2e}, "
          f"Inference Time: {test_metrics['avg_inference_time']:.4f}s, "
          f"Memory: {test_metrics['memory_used_mb']:.2f}MB, "
          f"GFLOPs: {test_metrics['gflops']:.4f}")
    
    return test_metrics
    
