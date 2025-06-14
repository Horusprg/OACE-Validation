import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import random
import numpy as np

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.DenseNet.densenet_architecture import generate_densenet_architecture
from utils.data_loader import get_cifar10_dataloaders
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model

def set_seed(seed):
    """Define a semente para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Configurações
    SEED = 42
    set_seed(SEED)
    
    # Parâmetros de treinamento
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # Parâmetros da DenseNet
    DENSENET_PARAMS = {
        'min_growth_rate': 12,
        'max_growth_rate': 48,
        'min_blocks': 3,
        'max_blocks': 5,
        'min_layers_per_block': 4,
        'max_layers_per_block': 16,
        'num_classes': 10,  # CIFAR-10
        'drop_rate': 0.2,
        'batch_norm': True
    }
    
    # Configuração do dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cria diretório para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', 'densenet_experiments', f'experiment_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Gera a arquitetura DenseNet
    model = generate_densenet_architecture(**DENSENET_PARAMS)
    print(f"Arquitetura DenseNet gerada com {sum(p.numel() for p in model.parameters())} parâmetros")
    
    # Configura otimizador e função de perda
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Carrega os dados
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(
        n_valid=0.2,
        batch_size=BATCH_SIZE,
        num_workers=2
    )
    
    # Treina o modelo
    print("\nIniciando treinamento...")
    training_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    # Avalia o modelo
    print("\nIniciando avaliação...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Salva os resultados
    results = {
        'training_metrics': training_metrics,
        'test_metrics': test_metrics,
        'model_config': DENSENET_PARAMS,
        'training_config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY
        }
    }
    
    # Salva o modelo
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    # Salva as métricas
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResultados salvos em: {results_dir}")

if __name__ == '__main__':
    main() 