#!/usr/bin/env python3
"""
Script de teste para treinamento especializado do MobileNet.
Este script testa a função specialized_training com os parâmetros otimizados.
"""

import torch
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import get_cifar10_dataloaders
from models.MobileNet.mobilenet_architecture import MobilenetParams
from models.MobileNet.mobilenet_warm_up import specialized_training

def test_mobilenet_specialized():
    """
    Testa o treinamento especializado do MobileNet.
    """
    print("="*70)
    print("TESTE DO TREINAMENTO ESPECIALIZADO MOBILENET")
    print("="*70)
    
    # Configuração do dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Dispositivo: {device}")
    
    # Carrega dados
    print(f"\n📊 Carregando dados CIFAR-10...")
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
    print(f"   • Classes: {len(classes)}")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    print(f"   • Test batches: {len(test_loader)}")
    
    # Parâmetros otimizados encontrados pelo algoritmo AFSA-GA-PSO
    optimized_params = {
        "num_classes": 10,
        "min_channels": 37,
        "max_channels": 1661,
        "dropout_rate": 0.0,
        "num_layers": 6,
        "batch_norm": True
    }
    
    print(f"\n🎯 Parâmetros otimizados:")
    for key, value in optimized_params.items():
        print(f"   • {key}: {value}")
    
    # Converte para MobilenetParams
    params = MobilenetParams(**optimized_params)
    
    # Configurações de treinamento para teste
    training_config = {
        'num_epochs': 100,  # Menos épocas para teste rápido
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'use_mixed_precision': True,
        'use_compile': True,
        'early_stopping_patience': 5,
        'save_best_model': True,
        'experiment_name': "mobilenet_test"
    }
    
    print(f"\n⚙️  Configuração de treinamento:")
    for key, value in training_config.items():
        print(f"   • {key}: {value}")
    
    try:
        # Executa treinamento especializado
        print(f"\n🚀 Iniciando treinamento especializado...")
        final_metrics = specialized_training(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=classes,
            params=params,
            device=device,
            **training_config
        )
        
        print(f"\n🏆 RESULTADOS DO TESTE:")
        print(f"   • Top-1 Accuracy: {final_metrics['top1_acc']:.2f}%")
        print(f"   • Top-5 Accuracy: {final_metrics['top5_acc']:.2f}%")
        print(f"   • F1-Score: {final_metrics['f1_macro']:.4f}")
        print(f"   • Loss: {final_metrics['loss']:.4f}")
        print(f"   • Parâmetros: {final_metrics['total_params']:.2e}")
        print(f"   • Tempo de Inferência: {final_metrics['avg_inference_time']:.4f}s")
        print(f"   • Memória: {final_metrics['memory_used_mb']:.2f} MB")
        print(f"   • GFLOPs: {final_metrics['gflops']:.4f}")
        
        
        return final_metrics
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Teste do treinamento especializado MobileNet')
    parser.add_argument('--full', action='store_true', help='Executa teste completo (20 épocas)')

    test_mobilenet_specialized()