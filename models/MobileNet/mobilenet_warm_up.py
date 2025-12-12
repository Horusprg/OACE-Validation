#!/usr/bin/env python3
"""
Módulo de treinamento e avaliação para MobileNet no CIFAR-10.

Este módulo fornece:
- warm_up_mobilenet: Treinamento rápido para testes iniciais
- train_mobilenet_specialized: Função de treinamento especializado com parâmetros configuráveis
"""

import os
import sys
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Adiciona o diretório raiz ao path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.MobileNet.mobilenet_architecture import generate_mobilenet_architecture, MobilenetParams
from utils.training_utils import train_model, get_optimized_scheduler
from utils.evaluate_utils import evaluate_model
from utils.data_loader import get_cifar10_dataloaders

def warm_up_mobilenet(
    train_loader,
    val_loader,
    test_loader,
    classes,
    num_epochs=3,
    device=None,
    params=None,
    learning_rate: float = 0.001
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = generate_mobilenet_architecture(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    results = {
        'experiment_id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'MobileNet',
        'mobilenet_params': params.dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes
    }

    results_dir = 'results/mobilenet_experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Resultados salvos em: {results_file}")

    #weights_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}_weights.pt')
    #torch.save(model.state_dict(), weights_file)
    #print(f"Pesos do modelo salvos em: {weights_file}")

    return test_metrics

# ============================================================================
# FUNÇÃO DE TREINAMENTO ESPECIALIZADO COM PARÂMETROS CONFIGURÁVEIS
# ============================================================================

def train_mobilenet_specialized(
    optimized_params=None,
    training_config=None,
    device=None
):
    """
    Treina o MobileNet com parâmetros otimizados de forma especializada.
    
    Esta função pode ser chamada diretamente ou via linha de comando.
    Utiliza parâmetros otimizados encontrados pelo algoritmo AFSA-GA-PSO.
    
    Returns:
        dict: Métricas finais do modelo treinado ou None em caso de erro
    """
    # Valores padrão para parâmetros otimizados (encontrados pelo AFSA-GA-PSO)
    if optimized_params is None:
        optimized_params = {
            "num_classes": 10,
            "min_channels": 32,
            "max_channels": 128,
            "dropout_rate": 0.5,
            "num_layers": 7,
            "batch_norm": True
        }
    
    # Valores padrão para configuração de treinamento
    if training_config is None:
        training_config = {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'use_mixed_precision': True,
            'use_compile': True,
            'early_stopping_patience': 5,
            'save_best_model': True,
            'experiment_name': "mobilenet_test"
        }
    
    print("="*70)
    print("TREINAMENTO ESPECIALIZADO MOBILENET")
    print("="*70)
    
    # Configuração do dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Dispositivo: {device}")
    
    # Carrega dados
    print(f"\n📊 Carregando dados CIFAR-10...")
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
    print(f"   • Classes: {len(classes)}")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    print(f"   • Test batches: {len(test_loader)}")
    
    print(f"\n🎯 Parâmetros otimizados:")
    for key, value in optimized_params.items():
        print(f"   • {key}: {value}")
    
    # Converte para MobilenetParams
    params = MobilenetParams(**optimized_params)
    
    print(f"\n⚙️  Configuração de treinamento:")
    for key, value in training_config.items():
        print(f"   • {key}: {value}")
    
    # Cria parâmetros e modelo
    params = MobilenetParams(**optimized_params)
    model = generate_mobilenet_architecture(params).to(device)
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  Arquitetura MobileNet gerada:")
    print(f"   • Total de parâmetros: {total_params:,}")
    print(f"   • Parâmetros treináveis: {trainable_params:,}")
    
    # Otimizador, critério e scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Scheduler configurável
    scheduler_type = training_config.get('scheduler_type', 'cosine')
    scheduler_kwargs = training_config.get('scheduler_kwargs', None)
    if scheduler_kwargs is None:
        scheduler_kwargs = {'T_max': training_config['num_epochs'], 'eta_min': 1e-6}
    
    scheduler = get_optimized_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        **scheduler_kwargs
    )
    
    try:
        # Treinamento usando train_model diretamente
        print(f"\n🚀 Iniciando treinamento especializado...")
        train_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=training_config['num_epochs'],
            device=device,
            use_mixed_precision=training_config.get('use_mixed_precision', True),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            early_stopping_patience=training_config.get('early_stopping_patience', 15),
            scheduler=scheduler,
            compile_model=training_config.get('use_compile', True)
        )
        
        # Avaliação final
        print(f"\n📊 Avaliação final especializada...")
        final_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Análise de performance
        print(f"\n📈 Análise de Performance:")
        print(f"   • Top-1 Accuracy: {final_metrics['top1_acc']:.2f}%")
        print(f"   • Top-5 Accuracy: {final_metrics['top5_acc']:.2f}%")
        print(f"   • F1-Score: {final_metrics['f1_macro']:.4f}")
        print(f"   • Parâmetros: {final_metrics['total_params']:.2e}")
        print(f"   • Tempo de Inferência: {final_metrics['avg_inference_time']:.4f}s")
        print(f"   • Memória: {final_metrics['memory_used_mb']:.2f} MB")
        print(f"   • GFLOPs: {final_metrics['gflops']:.4f}")
        
        # Salva resultados especializados
        experiment_id = str(uuid.uuid4())
        results = {
            'experiment_id': experiment_id,
            'experiment_name': training_config.get('experiment_name', 'mobilenet_specialized'),
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'model': 'MobileNet_Specialized',
            'mobilenet_params': params.model_dump() if hasattr(params, 'model_dump') else (params.dict() if hasattr(params, 'dict') else optimized_params),
            'training_config': {
                'num_epochs': training_config['num_epochs'],
                'learning_rate': training_config['learning_rate'],
                'weight_decay': training_config.get('weight_decay', 1e-4),
                'use_mixed_precision': training_config.get('use_mixed_precision', True),
                'use_compile': training_config.get('use_compile', True),
                'early_stopping_patience': training_config.get('early_stopping_patience', 15),
                'optimizer': 'AdamW',
                'scheduler': scheduler_type,
                'criterion': 'CrossEntropyLoss with Label Smoothing'
            },
            'model_info': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': final_metrics['memory_used_mb']
            },
            'train_metrics': train_metrics,
            'test_metrics': final_metrics,
            'classes': classes,
            'device': str(device)
        }
        
        # Cria diretório para resultados especializados
        results_dir = 'results/mobilenet_specialized'
        os.makedirs(results_dir, exist_ok=True)
        
        # Salva resultados
        results_file = os.path.join(results_dir, f'{training_config.get("experiment_name", "mobilenet_specialized")}_{experiment_id}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"\n💾 Resultados salvos em: {results_file}")
        
        # Salva modelo se solicitado
        """
        if training_config.get('save_best_model', True):
            weights_file = os.path.join(results_dir, f'{training_config.get("experiment_name", "mobilenet_specialized")}_{experiment_id}_weights.pt')
            
            # Limpa o state_dict antes de salvar (remove chaves extras)
            clean_state_dict = {}
            for key, value in model.state_dict().items():
                if not any(extra_key in key for extra_key in ['total_ops', 'total_params']):
                    clean_state_dict[key] = value
            
            torch.save({
                'model_state_dict': clean_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'params': params.model_dump() if hasattr(params, 'model_dump') else (params.dict() if hasattr(params, 'dict') else optimized_params),
                'test_metrics': final_metrics,
                'epoch': len(train_metrics)
            }, weights_file)
            print(f"💾 Modelo salvo em: {weights_file}")
        """
        # Log final
        print(f"\n✅ Treinamento especializado concluído!")
        print(f"   • Experimento: {training_config.get('experiment_name', 'mobilenet_specialized')}")
        print(f"   • ID: {experiment_id}")
        print(f"   • Melhor Top-1: {final_metrics['top1_acc']:.2f}%")
        print(f"   • Eficiência: {final_metrics['gflops']:.4f} GFLOPs")
        
        return final_metrics
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Permite execução direta do script via linha de comando.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Treinamento especializado MobileNet no CIFAR-10',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Exemplos de uso:
        python -m models.MobileNet.mobilenet_warm_up
        python -m models.MobileNet.mobilenet_warm_up --full
        """
    )
    parser.add_argument(
        '--full', 
        action='store_true', 
        help='Executa treinamento completo (100 épocas)'
    )
    
    args = parser.parse_args()
    
    # Parâmetros otimizados encontrados pelo algoritmo AFSA-GA-PSO
    optimized_params = {
        "num_classes": 10,
        "min_channels": 32,
        "max_channels": 128,
        "dropout_rate": 0.5,
        "num_layers": 7,
        "batch_norm": True
    }
    
    # Configurações de treinamento
    training_config = {
        'num_epochs': 2,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'use_mixed_precision': True,
        'use_compile': True,
        'early_stopping_patience': 5,
        'save_best_model': True,
        'experiment_name': "mobilenet_test",
        'scheduler_type': 'cosine',
        'scheduler_kwargs': {'T_max': 100, 'eta_min': 1e-6}
    }
    
    # Executa o treinamento especializado com os parâmetros
    train_mobilenet_specialized(
        optimized_params=optimized_params,
        training_config=training_config
    )
