#!/usr/bin/env python3
"""
Script principal para execução do algoritmo de otimização AFSA-GA-PSO.
"""

import torch
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.afsa_ga_pso import AFSAGAPSO
from utils.data_loader import get_cifar10_dataloaders, get_wildshapes_dataloaders

if __name__ == "__main__":
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    
    # Carregar os data loaders
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
    # train_loader, val_loader, test_loader, classes = get_wildshapes_dataloaders(batch_size=32)

    # Criar instância do otimizador híbrido
    optimizer = AFSAGAPSO(
        population_size=3,
        max_iter=2,  
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        lambda_param=0.5,
        afsa_params={'visual': 50, 'step': 8, 'try_times': 5, 'max_iter': 15},
        pso_params={"c1": 1.5, "c2": 1.5, "w": 0.7, "k": 3, "p": 2},
        ga_params={"initial_crossover_rate": 0.7, "initial_mutation_rate": 0.15, "tournament_size": 3, "max_iter": 3},
        architectures_to_optimize=['CNN', 'ResNet', 'EfficientNet', 'MobileNet']
    )

    # Executa a otimização
    best_architecture, best_params, best_fitness = optimizer.optimize()
    results = best_architecture, best_params, best_fitness
    
    print(f"\nMelhor arquitetura encontrada: {best_architecture}")
    print(f"Parâmetros da melhor arquitetura: {best_params}")
    print(f"Melhor valor de fitness (OACE): {best_fitness}")
    
    print("results: ", results)






