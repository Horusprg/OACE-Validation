#!/usr/bin/env python3
"""
Script principal para execução do algoritmo de otimização AFSA-GA-PSO.
(1) GPU_ID=3 nohup python3 -X utf8 -u -m main > results_6.log 2>&1 & disown
(2) GPU_ID=1 nohup python3 -X utf8 -u -m main > results_2.log 2>&1 & disown
"""

import os
import sys

# ===== CRÍTICO: Configurar CUDA_VISIBLE_DEVICES ANTES de importar PyTorch =====
gpu_id_env = os.environ.get('GPU_ID', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_env
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
print(f"🔒 CUDA_VISIBLE_DEVICES definido para: {gpu_id_env}")
# ==============================================================================

import torch

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.afsa_ga_pso import AFSAGAPSO
from utils.data_loader import get_cifar10_dataloaders, get_wildshapes_dataloaders

def setup_device():
    """
    Configura o dispositivo (device) para treinamento.
    Permite selecionar qual GPU usar através da variável de ambiente GPU_ID.
    Por padrão, usa GPU 0 se não especificado.
    
    IMPORTANTE: CUDA_VISIBLE_DEVICES já foi configurado antes de importar PyTorch,
    então PyTorch só vê a GPU selecionada como "cuda:0".
    
    Variáveis de ambiente:
        GPU_ID: Número da GPU física a ser usada (0, 1, 2, 3). Padrão: 0
    
    Returns:
        torch.device: Dispositivo configurado para uso
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA disponível: {torch.cuda.is_available()}")
        print(f"✓ Número de GPUs visíveis para PyTorch: {num_gpus}")
        
        # Lista todas as GPUs visíveis (deve ser apenas 1)
        for i in range(num_gpus):
            print(f"   GPU {i} (visível): {torch.cuda.get_device_name(i)}")
        
        # Com CUDA_VISIBLE_DEVICES configurado, sempre usamos cuda:0
        device = torch.device("cuda:0")
        print(f"✓ Usando device: {device}")
        print(f"✓ GPU física selecionada: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        print(f"💡 Para usar outra GPU, defina GPU_ID=0,1,2 ou 3 antes de executar")
        
        if num_gpus > 1:
            print(f"⚠️ AVISO: PyTorch vê {num_gpus} GPUs. CUDA_VISIBLE_DEVICES pode não estar funcionando!")
    else:
        print("⚠ CUDA não disponível. Usando CPU.")
        device = torch.device("cpu")

    return device

if __name__ == "__main__":
    # Configurar dispositivo
    device = setup_device()
    
    # Carregar os data loaders
    #train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=64, num_workers=2)
    # WildShapes2 usa imagens 224x224; batch menor reduz risco de OOM em arquiteturas grandes.
    train_loader, val_loader, test_loader, classes = get_wildshapes_dataloaders(batch_size=64, num_workers=2)

    # Criar instância do otimizador híbrido
    
    optimizer = AFSAGAPSO(
        population_size=20,
        max_iter=10,  
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        classes=classes,
        lambda_param=0.80,
        #afsa_params={'visual': 50, 'step': 8, 'try_times': 5, 'max_iter': 3},
        afsa_params={'visual': 40, 'step': 3, 'try_times': 2, 'max_iter': 8},
        pso_params={"c1": 1.5, "c2": 1.5, "w": 0.7, "k": 3, "p": 2},
        ga_params={"initial_crossover_rate": 0.70, "initial_mutation_rate": 0.30, "tournament_size": 3, "max_iter": 12},
        architectures_to_optimize=['CNN', 'ResNet', 'EfficientNet', 'MobileNet'],
        device=device  # Passa o device configurado
    )
    
    
    """
    optimizer = AFSAGAPSO(
        population_size=3,
        max_iter=2,  
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        classes=classes,
        lambda_param=0.80,
        #afsa_params={'visual': 50, 'step': 8, 'try_times': 5, 'max_iter': 3},
        afsa_params={'visual': 40, 'step': 3, 'try_times': 1, 'max_iter': 3},
        pso_params={"c1": 1.5, "c2": 1.5, "w": 0.7, "k": 3, "p": 2},
        ga_params={"initial_crossover_rate": 0.70, "initial_mutation_rate": 0.30, "tournament_size": 3, "max_iter": 3},
        architectures_to_optimize=['CNN'], #'ResNet', 'EfficientNet', 'MobileNet'],
        device=device  # Passa o device configurado
    )
    """
    # Executa a otimização
    best_architecture, best_params, best_fitness = optimizer.optimize()
    results = best_architecture, best_params, best_fitness
    
    print(f"\nMelhor arquitetura encontrada: {best_architecture}")
    print(f"Parâmetros da melhor arquitetura: {best_params}")
    print(f"Melhor valor de fitness (OACE): {best_fitness}")

    print("results: ", results)

# 290205 (wildshapes) e 142479 (cifar10)