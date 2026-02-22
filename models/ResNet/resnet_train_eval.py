#!/usr/bin/env python3
"""
Módulo de treinamento e avaliação para ResNet no CIFAR-10.

Este módulo fornece:
- warm_up_resnet: Treinamento rápido para testes iniciais
- train_resnet_specialized: Função de treinamento especializado com parâmetros configuráveis
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

from models.ResNet.resnet_architecture import ResNet, generate_resnet_architecture, ResNetParams
from utils.data_loader import get_cifar10_dataloaders
from utils.training_utils import train_model, get_optimized_scheduler
from utils.evaluate_utils import evaluate_model


def warm_up_resnet(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classes: list,
    num_epochs: int,
    device: torch.device,
    params: dict = None,
    learning_rate: float = 0.001
) -> dict:
    """
    Script warm_up para treinar e avaliar uma ResNet com parâmetros específicos no CIFAR-10
    
    Args:
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        test_loader: DataLoader para teste
        classes: Lista de classes
        num_epochs: Número de épocas para treinamento
        device: Dispositivo para execução
        params: Dicionário com parâmetros da arquitetura
        learning_rate: Taxa de aprendizado para o otimizador (padrão: 0.001)
        
    Returns:
        dict: Métricas de avaliação da rede
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cria parâmetros ResNetParams a partir do dicionário
    if params is None:
        resnet_params = ResNetParams()
    elif isinstance(params, dict):
        resnet_params = ResNetParams(**params)
    elif isinstance(params, ResNetParams):
        resnet_params = params
    else:
        raise ValueError(f"Tipo de parâmetros não suportado: {type(params)}")
    
    # Gera a arquitetura ResNet com os parâmetros
    model = generate_resnet_architecture(resnet_params).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Treina o modelo
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        use_mixed_precision=True,
    )

    # Avalia o modelo
    test_metrics = evaluate_model(
        model=model, 
        test_loader=test_loader, 
        criterion=criterion, 
        device=device
    )
    
    results = {
        'experiment_id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'ResNet',
        'resnet_params': resnet_params.model_dump() if hasattr(resnet_params, 'model_dump') else (resnet_params.dict() if hasattr(resnet_params, 'dict') else str(resnet_params)),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes
    }

    results_dir = 'results/resnet_experiments'
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

def train_resnet_specialized(
    optimized_params=None,
    training_config=None,
    device=None
):
    """
    Treina o ResNet com parâmetros otimizados de forma especializada.
    
    Esta função pode ser chamada diretamente ou via linha de comando.
    Utiliza parâmetros otimizados encontrados pelo algoritmo AFSA-GA-PSO.
    
    Args:
        optimized_params (dict, optional): Parâmetros da arquitetura ResNet.
            Se None, usa valores padrão otimizados pelo AFSA-GA-PSO.
            Formato esperado:
            {
                "num_classes": int,
                "min_channels": int,
                "max_channels": int,
                "dropout_rate": float,
                "num_layers": int,
                "batch_norm": bool
            }
        
        training_config (dict, optional): Configurações de treinamento.
            Se None, usa valores padrão otimizados.
            Formato esperado:
            {
                'num_epochs': int,
                'learning_rate': float,
                'weight_decay': float,
                'use_mixed_precision': bool,
                'use_compile': bool,
                'early_stopping_patience': int,
                'gradient_accumulation_steps': int,
                'max_grad_norm': float,
                'scheduler_type': str,
                'scheduler_kwargs': dict,
                'save_best_model': bool,
                'experiment_name': str
            }
        
        device (torch.device, optional): Dispositivo para treinamento.
            Se None, detecta automaticamente (cuda se disponível, senão cpu).
    
    Returns:
        dict: Métricas finais do modelo treinado ou None em caso de erro
    """
    # Valores padrão para parâmetros otimizados (encontrados pelo AFSA-GA-PSO)
    if optimized_params is None:
        optimized_params = {
            "num_classes": 10,
            "min_channels": 40,
            "max_channels": 64,
            "dropout_rate": 0.0,
            "num_layers": 28,
            "batch_norm": True,
        }
    
    # Valores padrão para configuração de treinamento
    if training_config is None:
        training_config = {
            "num_epochs": 150,
            "learning_rate": 0.000472,
            "weight_decay": 1e-4,
            "use_mixed_precision": True,
            "use_compile": True,
            "early_stopping_patience": 10,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "scheduler_type": "cosine",  # 'cosine' | 'step' | 'plateau'
            "scheduler_kwargs": {"T_max": 100, "eta_min": 1e-6},
            "save_best_model": True,
            "experiment_name": "resnet_full",
        }
    
    print("=" * 70)
    print("TREINAMENTO ESPECIALIZADO RESNET")
    print("=" * 70)
    
    # Configuração do dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Dispositivo: {device}")
    
    # Carrega dados
    print("\n📊 Carregando dados CIFAR-10...")
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
    print(f"   • Classes: {len(classes)}")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    print(f"   • Test batches: {len(test_loader)}")
    
    print("\n🎯 Parâmetros da ResNet:")
    for key, value in optimized_params.items():
        print(f"   • {key}: {value}")
    
    # Cria parâmetros e modelo
    params = ResNetParams(**optimized_params)
    model = generate_resnet_architecture(params).to(device)
    
    print("\n⚙️  Configuração de treinamento:")
    for key, value in training_config.items():
        print(f"   • {key}: {value}")
    
    # Otimizador, critério e scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 1e-4)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = get_optimized_scheduler(
        optimizer,
        scheduler_type=training_config["scheduler_type"],
        **training_config.get("scheduler_kwargs", {}),
    )
    
    try:
        # Treinamento
        print("\n🚀 Iniciando treinamento...")
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=training_config["num_epochs"],
            device=device,
            use_mixed_precision=training_config["use_mixed_precision"],
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            early_stopping_patience=training_config["early_stopping_patience"],
            scheduler=scheduler,
            compile_model=training_config["use_compile"],
        )
        
        # Avaliação final
        print("\n🧪 Avaliando no conjunto de teste...")
        final_metrics = evaluate_model(
            model=model, 
            test_loader=test_loader, 
            criterion=criterion, 
            device=device
        )
        
        # Resultado estruturado
        results = {
            "experiment_id": str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model": "ResNet",
            "resnet_params": params.model_dump() if hasattr(params, 'model_dump') else (params.dict() if hasattr(params, 'dict') else optimized_params),
            "training_config": training_config,
            "history": metrics,
            "test_metrics": final_metrics,
            "classes": classes,
        }
        
        results_dir = os.path.join("results", "resnet_experiments")
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"experiment_{results['experiment_id']}.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Resultados salvos em: {results_file}")
        
        # Salva modelo se solicitado
        if training_config.get("save_best_model", False):
            weights_file = os.path.join(results_dir, f"{training_config['experiment_name']}_{results['experiment_id']}_weights.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'params': params.model_dump() if hasattr(params, 'model_dump') else (params.dict() if hasattr(params, 'dict') else optimized_params),
                'test_metrics': final_metrics,
                'epoch': len(metrics)
            }, weights_file)
            print(f"💾 Modelo salvo em: {weights_file}")
        
        print("\n🏆 RESULTADOS FINAIS:")
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
        print(f"\n❌ Erro durante o treinamento: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Permite execução direta do script via linha de comando.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Treinamento especializado ResNet no CIFAR-10',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Exemplos de uso:
        python -m models.ResNet.resnet_train_eval
        python -m models.ResNet.resnet_train_eval --full
        """
    )
    parser.add_argument(
        '--full', 
        action='store_true', 
        help='Executa treinamento completo (150 épocas)'
    )
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--wd", type=float, default=None, help="Weight decay")
    parser.add_argument("--mixed", action="store_true", help="Usar mixed precision")
    parser.add_argument("--compile", action="store_true", help="Compilar modelo")
    
    args = parser.parse_args()
    
    # Parâmetros otimizados encontrados pelo algoritmo AFSA-GA-PSO
    optimized_params = {
        "num_classes": 10,
        "min_channels": 40,
        "max_channels": 64,
        "dropout_rate": 0.0,
        "num_layers": 28,
        "batch_norm": True,
    }
    
    # Configurações de treinamento
    training_config = {
        "num_epochs": 1,
        "learning_rate": 0.000472,
        "weight_decay": 1e-4,
        "use_mixed_precision": True,
        "use_compile": True,
        "early_stopping_patience": 10,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "scheduler_type": "cosine",
        "scheduler_kwargs": {"T_max": 100, "eta_min": 1e-6},
        "save_best_model": True,
        "experiment_name": "resnet_full",
    }
    
    # Ajusta configurações via CLI se fornecidas
    if args.epochs:
        training_config["num_epochs"] = args.epochs
    if args.lr:
        training_config["learning_rate"] = args.lr
    if args.wd:
        training_config["weight_decay"] = args.wd
    if args.mixed:
        training_config["use_mixed_precision"] = True
    if args.compile:
        training_config["use_compile"] = True
    
    # Executa o treinamento especializado com os parâmetros
    train_resnet_specialized(
        optimized_params=optimized_params,
        training_config=training_config
    )
