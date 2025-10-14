#!/usr/bin/env python3
"""
Script de teste para demonstrar o sistema de logging aprimorado do algoritmo híbrido AFSA-GA-PSO.

Este script executa o algoritmo híbrido com logging detalhado e demonstra como as variáveis
importantes de cada geração e iteração são capturadas e armazenadas.
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.afsa_ga_pso import AFSAGAPSO
from utils.data_loader import get_cifar10_dataloaders
from utils.optimization_logger import OptimizationLogger

def test_enhanced_logging():
    """
    Testa o sistema de logging aprimorado com o algoritmo híbrido AFSA-GA-PSO.
    """
    print("="*80)
    print("TESTE DO SISTEMA DE LOGGING APRIMORADO - AFSA-GA-PSO")
    print("="*80)
    
    # Verifica CUDA
    print(f"🔧 CUDA disponível: {torch.cuda.is_available()}")
    print(f"🔧 Número de GPUs: {torch.cuda.device_count()}")
    
    # Carrega os dados
    print("\n📊 Carregando dados CIFAR-10...")
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
    print(f"✅ Dados carregados: {len(classes)} classes")
    
    # Configuração do experimento (parâmetros reduzidos para teste rápido)
    experiment_config = {
        "population_size": 3,  # Reduzido para teste rápido
        "max_iter": 2,         # Reduzido para teste rápido
        "lambda_param": 0.5,
        "afsa_params": {
            'visual': 0.5, 
            'step': 0.1, 
            'try_times': 3, 
            'max_iter': 2  # Reduzido para teste rápido
        },
        "pso_params": {
            'c1': 1.5,
            'c2': 1.5,
            'w': 0.7,
            'k': 3,
            'p': 2
        },
        "ga_params": {
            'initial_crossover_rate': 0.8,
            'initial_mutation_rate': 0.15,
            'tournament_size': 3,
            'max_iter': 2  # Reduzido para teste rápido
        },
        "architectures_to_optimize": ['CNN']  # Apenas CNN para teste rápido
    }
    
    print(f"\n⚙️  Configuração do experimento:")
    for key, value in experiment_config.items():
        print(f"   • {key}: {value}")
    
    # Cria o otimizador híbrido
    print(f"\n🚀 Inicializando otimizador híbrido AFSA-GA-PSO...")
    optimizer = AFSAGAPSO(
        population_size=experiment_config["population_size"],
        max_iter=experiment_config["max_iter"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        lambda_param=experiment_config["lambda_param"],
        afsa_params=experiment_config["afsa_params"],
        pso_params=experiment_config["pso_params"],
        ga_params=experiment_config["ga_params"],
        architectures_to_optimize=experiment_config["architectures_to_optimize"],
        log_dir="results"
    )
    
    print(f"✅ Otimizador inicializado com logging aprimorado")
    
    # Executa a otimização
    print(f"\n🎯 Iniciando otimização híbrida...")
    start_time = datetime.now()
    
    try:
        best_architecture, best_params, best_fitness = optimizer.optimize()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 OTIMIZAÇÃO CONCLUÍDA!")
        print(f"⏱️  Tempo total: {duration:.2f} segundos")
        print(f"🏆 Melhor arquitetura: {best_architecture}")
        print(f"📊 Melhor score OACE: {best_fitness:.6f}")
        print(f"⚙️  Melhores parâmetros: {best_params}")
        
        # Análise dos logs gerados
        print(f"\n📋 ANÁLISE DOS LOGS GERADOS:")
        print(f"   • Diretório do experimento: {optimizer.logger.experiment_dir}")
        print(f"   • Total de iterações registradas: {len(optimizer.logger.log_data['iterations'])}")
        print(f"   • Total de avaliações: {optimizer.logger.log_data['optimization_summary']['total_evaluations']}")
        print(f"   • Arquiteturas únicas testadas: {optimizer.logger.log_data['optimization_summary']['total_architectures_tested']}")
        
        # Estatísticas de cache
        cache_stats = optimizer.logger.cache_stats
        total_evaluations = cache_stats['total_evaluations']
        cache_efficiency = (cache_stats['cache_hits'] / total_evaluations * 100) if total_evaluations > 0 else 0
        print(f"   • Eficiência do cache: {cache_efficiency:.1f}%")
        print(f"   • Cache hits: {cache_stats['cache_hits']}")
        print(f"   • Cache misses: {cache_stats['cache_misses']}")
        
        # Análise por fase
        print(f"\n📊 ANÁLISE POR FASE:")
        phases = {}
        for iteration in optimizer.logger.log_data['iterations']:
            phase = iteration['phase']
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(iteration['best_fitness'])
        
        for phase, fitness_values in phases.items():
            if fitness_values:
                print(f"   • {phase}:")
                print(f"     - Iterações: {len(fitness_values)}")
                print(f"     - Melhor fitness: {max(fitness_values):.6f}")
                print(f"     - Pior fitness: {min(fitness_values):.6f}")
                print(f"     - Fitness médio: {np.mean(fitness_values):.6f}")
        
        # Demonstração dos dados específicos dos algoritmos
        print(f"\n🔍 DADOS ESPECÍFICOS DOS ALGORITMOS:")
        for iteration in optimizer.logger.log_data['iterations']:
            phase = iteration['phase']
            algorithm_data = iteration.get('algorithm_specific_data', {})
            
            if algorithm_data:
                print(f"\n   📌 Iteração {iteration['iteration']} - Fase {phase}:")
                
                # Dados do AFSA
                if 'visual' in algorithm_data:
                    print(f"      🐟 AFSA - Visual: {algorithm_data.get('visual', 'N/A')}")
                    print(f"      🐟 AFSA - Step: {algorithm_data.get('step', 'N/A')}")
                    print(f"      🐟 AFSA - Try Times: {algorithm_data.get('try_times', 'N/A')}")
                
                # Dados do PSO
                if 'inertia_weight' in algorithm_data:
                    print(f"      🐝 PSO - Inércia: {algorithm_data.get('inertia_weight', 'N/A')}")
                    print(f"      🐝 PSO - Coef. Cognitivo: {algorithm_data.get('cognitive_coeff', 'N/A')}")
                    print(f"      🐝 PSO - Coef. Social: {algorithm_data.get('social_coeff', 'N/A')}")
                
                # Dados do GA
                if 'crossover_rate' in algorithm_data:
                    print(f"      🧬 GA - Taxa Crossover: {algorithm_data.get('crossover_rate', 'N/A')}")
                    print(f"      🧬 GA - Taxa Mutação: {algorithm_data.get('mutation_rate', 'N/A')}")
                    print(f"      🧬 GA - Pressão Seleção: {algorithm_data.get('selection_pressure', 'N/A')}")
                
                # Métricas de diversidade
                diversity_metrics = algorithm_data.get('diversity_metrics', {})
                if diversity_metrics:
                    print(f"      📊 Diversidade: {diversity_metrics.get('diversity', 'N/A'):.4f}")
                    print(f"      📊 Spread: {diversity_metrics.get('spread', 'N/A'):.4f}")
                    print(f"      📊 Convergência: {diversity_metrics.get('convergence', 'N/A'):.4f}")
        
        print(f"\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        print(f"📁 Verifique os arquivos de log em: {optimizer.logger.experiment_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO durante a otimização: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Iniciando teste do sistema de logging aprimorado...")
    success = test_enhanced_logging()
    
    if success:
        print("\n🎉 Teste concluído com sucesso!")
        sys.exit(0)
    else:
        print("\n💥 Teste falhou!")
        sys.exit(1)




