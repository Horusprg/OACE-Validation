import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int, device: torch.device
                ) -> List[Dict[str, float]]:
    """
    Treina um modelo no CIFAR-10 de forma genérica para warm-up.

    Args:
        model: Modelo PyTorch (herda de nn.Module).
        train_loader: DataLoader para dados de treinamento.
        val_loader: DataLoader para dados de validação.
        criterion: Função de perda (ex.: CrossEntropyLoss).
        optimizer: Otimizador (ex.: Adam).
        num_epochs: Número de épocas.
        device: Dispositivo (CPU ou GPU).

    Returns:
        List[Dict[str, float]]: Lista de dicionários com métricas por época
            (train_loss, train_acc, valid_loss, valid_acc).
    """
    model.to(device)
    metrics = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
                inputs = inputs.view(inputs.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1) # _,predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        model.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
                    inputs = inputs.view(inputs.size(0), -1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1) # _,predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / train_total,
            'train_acc': 100.0 * train_correct / train_total,
            'valid_loss': valid_loss / valid_total,
            'valid_acc': 100.0 * valid_correct / valid_total
        }
        metrics.append(epoch_metrics)
        
        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {epoch_metrics['train_loss']:.4f}, "
              f"Train Acc: {epoch_metrics['train_acc']:.2f}%, "
              f"Valid Loss: {epoch_metrics['valid_loss']:.4f}, "
              f"Valid Acc: {epoch_metrics['valid_acc']:.2f}%")
        
    return metrics
                    
                    
        
            
    