#!/usr/bin/env python3
"""
Script principal para execução do algoritmo de otimização AFSA-GA-PSO.
(1) GPU_ID=3 nohup python3 -X utf8 -u -m main > results_6.log 2>&1 & disown
(2) GPU_ID=1 nohup python3 -X utf8 -u -m main > results_2.log 2>&1 & disown
"""

import torch
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.afsa_ga_pso import AFSAGAPSO
from utils.data_loader import get_cifar10_dataloaders, get_wildshapes_dataloaders

def setup_device():
    """
    Configura o dispositivo (device) para treinamento.
    Permite selecionar qual GPU usar através da variável de ambiente GPU_ID.
    Por padrão, usa GPU 0 se não especificado.
    Para usar múltiplas GPUs, defina USE_DATAPARALLEL=1 (pode causar erros NCCL).
    
    Variáveis de ambiente:
        GPU_ID: Número da GPU a ser usada (0, 1, 2, 3). Padrão: 0
        USE_DATAPARALLEL: Se '1', 'true' ou 'yes', habilita DataParallel
    
    Returns:
        torch.device: Dispositivo configurado para uso
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA disponível: {torch.cuda.is_available()}")
        print(f"✓ Número de GPUs detectadas: {num_gpus}")
        
        # Lista todas as GPUs disponíveis
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Obtém o ID da GPU da variável de ambiente
        gpu_id_env = os.environ.get('GPU_ID', '0')
        try:
            gpu_id = int(gpu_id_env)
        except ValueError:
            print(f"⚠ Valor inválido para GPU_ID: '{gpu_id_env}'. Usando GPU 0.")
            gpu_id = 0
        
        # Valida se a GPU selecionada existe
        if gpu_id < 0 or gpu_id >= num_gpus:
            print(f"⚠ GPU {gpu_id} não disponível. GPUs disponíveis: 0-{num_gpus-1}")
            print(f"💡 Usando GPU 0 por padrão.")
            gpu_id = 0
        
        # Configura o device para a GPU selecionada
        device = torch.device(f"cuda:{gpu_id}")
        print(f"✓ GPU selecionada: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        print(f"💡 Para usar outra GPU, defina GPU_ID=0,1,2 ou 3 antes de executar")
        
        # Verifica se DataParallel está habilitado
        if num_gpus > 1:
            use_dp = os.environ.get('USE_DATAPARALLEL', '0').lower() in ('1', 'true', 'yes')
            if use_dp:
                print(f"⚠ DataParallel habilitado via USE_DATAPARALLEL=1 (pode causar erros NCCL)")
            else:
                print(f"💡 Usando GPU única. Para usar múltiplas GPUs, defina USE_DATAPARALLEL=1")
    else:
        print("⚠ CUDA não disponível. Usando CPU.")
        device = torch.device("cpu")

    return device

if __name__ == "__main__":
    # Configurar dispositivo
    device = setup_device()
    
    # Carregar os data loaders
    #train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=64, num_workers=2)
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
        afsa_params={'visual': 40, 'step': 3, 'try_times': 1, 'max_iter': 8},
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






