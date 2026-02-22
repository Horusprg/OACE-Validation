#!/usr/bin/env python3
# 220196
"""
Script principal com suporte a Checkpoint para o algoritmo AFSA-GA-PSO.

Este script usa o wrapper AFSAGAPSOWithCheckpoint que permite:
- Salvar checkpoints automaticamente durante a otimização
- Retomar de onde parou em caso de interrupção

Uso:
    # Primeira execução (inicia do zero):
    GPU_ID=3 nohup python3 -X utf8 -u -m main_checkpoint > results_checkpoint.log 2>&1 & disown

    # Retomar execução anterior:
    GPU_ID=3 RESUME=1 nohup python3 -X utf8 -u -m main_checkpoint > results_checkpoint.log 2>&1 & disown

    # Retomar de checkpoint específico:
    GPU_ID=3 RESUME=/path/to/checkpoint.ckpt nohup python3 -X utf8 -u -m main_checkpoint > results_checkpoint.log 2>&1 & disown

Variáveis de Ambiente:
    GPU_ID: ID da GPU a usar (0, 1, 2, 3). Padrão: 0
    RESUME: Se "1" ou "true", retoma do último checkpoint. 
            Se for um caminho, retoma desse checkpoint específico.
    CHECKPOINT_DIR: Diretório para checkpoints. Padrão: "checkpoints"
    CHECKPOINT_INTERVAL: Intervalo entre checkpoints. Padrão: 1
    EXPERIMENT_ID: ID do experimento (gerado automaticamente se não fornecido)

## Retomar experimento 1
GPU_ID=2 CHECKPOINT_DIR=checkpoints_exp RESUME=1 nohup python3 -X utf8 -u -m main_checkpoint > results_exp1.log 2>&1 & disown

# Retomar experimento 2
GPU_ID=3 CHECKPOINT_DIR=checkpoints_exp2 RESUME=1 nohup python3 -X utf8 -u -m main_checkpoint > results_exp2.log 2>&1 & disown

"""
#220196
import os
import sys

# ===== CRÍTICO: Configurar CUDA_VISIBLE_DEVICES ANTES de importar PyTorch =====
gpu_id_env = os.environ.get('GPU_ID', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_env
print(f"🔒 CUDA_VISIBLE_DEVICES definido para: {gpu_id_env}")
# ==============================================================================

import torch

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.afsa_ga_pso_checkpoint import AFSAGAPSOWithCheckpoint
from utils.data_loader import get_cifar10_dataloaders, get_wildshapes_dataloaders


def setup_device():
    """
    Configura o dispositivo (device) para treinamento.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA disponível: {torch.cuda.is_available()}")
        print(f"✓ Número de GPUs visíveis para PyTorch: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"   GPU {i} (visível): {torch.cuda.get_device_name(i)}")
        
        device = torch.device("cuda:0")
        print(f"✓ Usando device: {device}")
        print(f"✓ GPU física selecionada: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        
        if num_gpus > 1:
            print(f"⚠️ AVISO: PyTorch vê {num_gpus} GPUs. CUDA_VISIBLE_DEVICES pode não estar funcionando!")
    else:
        print("⚠ CUDA não disponível. Usando CPU.")
        device = torch.device("cpu")

    return device


def get_resume_config():
    """
    Obtém configuração de retomada das variáveis de ambiente.
    
    Returns:
        Union[bool, str]: False se não retomar, True para retomar do último,
                          ou caminho específico do checkpoint
    """
    resume_env = os.environ.get('RESUME', '').lower()
    
    if resume_env in ('1', 'true', 'yes'):
        return True
    elif resume_env and os.path.exists(resume_env):
        return resume_env
    elif resume_env and not resume_env in ('0', 'false', 'no', ''):
        # Pode ser um caminho que não existe ainda
        print(f"⚠️ Checkpoint especificado não encontrado: {resume_env}")
        print(f"   Tentando buscar checkpoints existentes...")
        return True
    
    return False


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 OTIMIZAÇÃO HÍBRIDA AFSA-GA-PSO COM CHECKPOINT")
    print("=" * 80)
    
    # Configurar dispositivo
    device = setup_device()
    
    # Configurações de checkpoint
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
    checkpoint_interval = int(os.environ.get('CHECKPOINT_INTERVAL', '1'))
    experiment_id = os.environ.get('EXPERIMENT_ID', None)
    resume_from_checkpoint = get_resume_config()
    
    print(f"\n📋 Configuração de Checkpoint:")
    print(f"   • Diretório: {checkpoint_dir}")
    print(f"   • Intervalo: {checkpoint_interval}")
    print(f"   • Experiment ID: {experiment_id or 'auto-gerado'}")
    print(f"   • Retomar: {resume_from_checkpoint}")
    
    # Carregar os data loaders
    print(f"\n📂 Carregando dataset WildShapes...")
    #train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=128, num_workers=2)
    train_loader, val_loader, test_loader, classes = get_wildshapes_dataloaders(
        batch_size=64, 
        num_workers=4
    )
    print(f"   ✓ Dataset carregado com {len(classes)} classes")

    # Criar instância do otimizador COM CHECKPOINT
    print(f"\n🔧 Inicializando otimizador...")
    
    optimizer = AFSAGAPSOWithCheckpoint(
        # Parâmetros do algoritmo (mesmos do original)
        population_size=20,
        max_iter=10,  
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        classes=classes,
        lambda_param=0.85,
        afsa_params={'visual': 40, 'step': 3, 'try_times': 2, 'max_iter': 8},
        pso_params={"c1": 1.5, "c2": 1.5, "w": 0.7, "k": 3, "p": 2},
        ga_params={"initial_crossover_rate": 0.70, "initial_mutation_rate": 0.30, "tournament_size": 3, "max_iter": 12},
        architectures_to_optimize=['CNN', 'ResNet', 'EfficientNet', 'MobileNet'],
        device=device,
        
        # Parâmetros de checkpoint (NOVOS)
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from_checkpoint,
        experiment_id=experiment_id,
        max_checkpoints=3  # Mantém os 3 checkpoints mais recentes por fase
    )
    
    # Executa a otimização
    print(f"\n🏁 Iniciando otimização...")
    best_architecture, best_params, best_fitness = optimizer.optimize()
    results = best_architecture, best_params, best_fitness
    
    print("\n" + "=" * 80)
    print("🏆 RESULTADOS FINAIS")
    print("=" * 80)
    print(f"\nMelhor arquitetura encontrada: {best_architecture}")
    print(f"Parâmetros da melhor arquitetura: {best_params}")
    print(f"Melhor valor de fitness (OACE): {best_fitness}")
    print(f"\nresults: {results}")
    print("=" * 80)
