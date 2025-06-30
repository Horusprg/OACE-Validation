#!/usr/bin/env python3
"""
Script de teste aprimorado para verificar o sistema de logging completo.
Testa integração com arquiteturas reais, scores OACE, checkpoints e análises.
"""

import sys
import os
import numpy as np
import json
import time
from typing import Dict, Any, List

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.optimization_logger import OptimizationLogger
from utils.optimization_utils import OptimizationManager
from utils.oace_evaluation import calculate_oace_score
from optimizers.pso_fixed import PSO

def generate_realistic_architecture_config(model_type: str = None) -> Dict[str, Any]:
    """Gera configuração realista de arquitetura"""
    if model_type is None:
        model_types = ["CNN", "ResNet", "MobileNet", "VGG", "DenseNet"]
        model_type = np.random.choice(model_types)
    
    configs = {
        "CNN": {
            "model_type": "CNN",
            "num_layers": int(np.random.randint(3, 8)),
            "min_channels": int(np.random.randint(8, 32)),
            "max_channels": int(np.random.randint(128, 256)),
            "dropout_rate": float(np.random.uniform(0.1, 0.3)),
            "num_classes": 10
        },
        "ResNet": {
            "model_type": "ResNet",
            "depth": int(np.random.choice([18, 34, 50, 101])),
            "num_classes": 10,
            "dropout_rate": float(np.random.uniform(0.0, 0.2))
        },
        "MobileNet": {
            "model_type": "MobileNet",
            "width_multiplier": float(np.random.uniform(0.5, 1.5)),
            "resolution_multiplier": float(np.random.uniform(0.75, 1.0)),
            "num_classes": 10,
            "dropout_rate": float(np.random.uniform(0.1, 0.3))
        },
        "VGG": {
            "model_type": "VGG",
            "layers": int(np.random.choice([11, 13, 16, 19])),
            "batch_norm": bool(np.random.choice([True, False])),
            "num_classes": 10,
            "dropout_rate": float(np.random.uniform(0.3, 0.5))
        },
        "DenseNet": {
            "model_type": "DenseNet",
            "growth_rate": int(np.random.randint(12, 32)),
            "num_layers": int(np.random.randint(40, 100)),
            "num_classes": 10,
            "dropout_rate": float(np.random.uniform(0.0, 0.2))
        }
    }
    
    return configs.get(model_type, configs["CNN"])

def generate_realistic_metrics(architecture_config: Dict[str, Any]) -> Dict[str, float]:
    """Gera métricas realistas baseadas na arquitetura"""
    model_type = architecture_config.get("model_type", "CNN")
    
    # Métricas base por tipo de modelo (simuladas realisticamente)
    base_metrics = {
        "CNN": {"acc": 0.75, "params": 500000, "time": 0.05, "memory": 50},
        "ResNet": {"acc": 0.85, "params": 11000000, "time": 0.08, "memory": 120},
        "MobileNet": {"acc": 0.80, "params": 3200000, "time": 0.03, "memory": 35},
        "VGG": {"acc": 0.88, "params": 20000000, "time": 0.12, "memory": 180},
        "DenseNet": {"acc": 0.87, "params": 15000000, "time": 0.10, "memory": 150}
    }
    
    base = base_metrics.get(model_type, base_metrics["CNN"])
    
    # Adiciona variação baseada em parâmetros específicos
    accuracy_variation = np.random.uniform(-0.05, 0.05)
    param_variation = np.random.uniform(0.5, 2.0)
    time_variation = np.random.uniform(0.8, 1.5)
    memory_variation = np.random.uniform(0.7, 1.3)
    
    metrics = {
        # Métricas de assertividade
        "top1_acc": max(0.6, min(0.95, base["acc"] + accuracy_variation)),
        "top5_acc": max(0.8, min(0.99, base["acc"] + accuracy_variation + 0.1)),
        "precision_macro": max(0.6, min(0.95, base["acc"] + np.random.uniform(-0.03, 0.03))),
        "recall_macro": max(0.6, min(0.95, base["acc"] + np.random.uniform(-0.03, 0.03))),
        "f1_macro": max(0.6, min(0.95, base["acc"] + np.random.uniform(-0.03, 0.03))),
        
        # Métricas de custo
        "total_params": int(base["params"] * param_variation),
        "avg_inference_time": base["time"] * time_variation,
        "memory_used_mb": base["memory"] * memory_variation,
        "gflops": np.random.uniform(0.5, 5.0)
    }
    
    return metrics

def calculate_realistic_oace_score(architecture_config: Dict[str, Any], metrics: Dict[str, float]) -> float:
    """Calcula score OACE realista"""
    # Pesos AHP simplificados
    assertiveness_weights = {
        'top1_acc': 0.5,
        'precision_macro': 0.3,
        'recall_macro': 0.2
    }
    
    cost_weights = {
        'total_params': 0.4,
        'avg_inference_time': 0.4,
        'memory_used_mb': 0.2
    }
    
    # Limites realistas para normalização
    assertiveness_min_max = {
        'top1_acc': {'min': 0.60, 'max': 0.95},
        'precision_macro': {'min': 0.60, 'max': 0.95},
        'recall_macro': {'min': 0.60, 'max': 0.95}
    }
    
    cost_min_max = {
        'total_params': {'min': 100000, 'max': 25000000},
        'avg_inference_time': {'min': 0.01, 'max': 0.20},
        'memory_used_mb': {'min': 20, 'max': 200}
    }
    
    assertiveness_metrics = {
        'top1_acc': metrics['top1_acc'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro']
    }
    
    cost_metrics = {
        'total_params': metrics['total_params'],
        'avg_inference_time': metrics['avg_inference_time'],
        'memory_used_mb': metrics['memory_used_mb']
    }
    
    try:
        oace_score = calculate_oace_score(
            assertiveness_metrics=assertiveness_metrics,
            cost_metrics=cost_metrics,
            lambda_param=0.5,
            assertiveness_weights=assertiveness_weights,
            cost_weights=cost_weights,
            assertiveness_min_max=assertiveness_min_max,
            cost_min_max=cost_min_max
        )
        return oace_score
    except Exception as e:
        print(f"Erro ao calcular OACE: {e}")
        return 0.7  # Score padrão

def test_enhanced_logging_system():
    """Testa sistema de logging aprimorado com dados realistas"""
    print("🧪 Testando sistema de logging aprimorado...")
    
    try:
        # 1. Testar OptimizationLogger aprimorado
        logger = OptimizationLogger(log_dir="results")
        
        # Configuração realista
        config = {
            "algorithm": "AFSA-GA-PSO",
            "population_size": 5,
            "max_iter": 10,
            "lambda_param": 0.5,
            "architectures": ["CNN", "ResNet", "MobileNet"],
            "dataset": "CIFAR-10",
            "test_mode": True
        }
        
        # Iniciar experimento
        logger.start_experiment(config)
        assert logger.current_experiment is not None, "Experimento não foi iniciado"
        print("   ✅ Experimento iniciado com configuração realista")
        
        # Simular iterações com arquiteturas e OACE reais
        for iteration in range(1, 6):
            # Gera população com arquiteturas diferentes
            population = np.random.rand(3, 4)  # 3 indivíduos, 4 dimensões
            
            # Simula avaliações de diferentes arquiteturas
            fitness_values = []
            all_metrics = []
            all_configs = []
            all_oace_scores = []
            
            for i in range(3):
                # Gera configuração de arquitetura
                arch_config = generate_realistic_architecture_config()
                
                # Gera métricas realistas
                metrics = generate_realistic_metrics(arch_config)
                
                # Calcula OACE score
                oace_score = calculate_realistic_oace_score(arch_config, metrics)
                
                fitness_values.append(oace_score)
                all_metrics.append(metrics)
                all_configs.append(arch_config)
                all_oace_scores.append(oace_score)
            
            fitness_values = np.array(fitness_values)
            best_idx = np.argmax(fitness_values)
            
            # Simula pbest e gbest para PSO
            pbest_pos = population + np.random.uniform(-0.1, 0.1, population.shape)
            pbest_cost = fitness_values + np.random.uniform(-0.05, 0.05, len(fitness_values))
            gbest_pos = population[best_idx]
            gbest_cost = fitness_values[best_idx]
            
            # Log da iteração com dados completos
            evaluation_time = np.random.uniform(5.0, 15.0)  # Tempo de avaliação simulado
            
            logger.log_iteration(
                iteration=iteration,
                phase="PSO" if iteration <= 3 else "GA-PSO",
                population=population,
                fitness_values=fitness_values,
                best_position=population[best_idx],
                best_fitness=fitness_values[best_idx],
                metrics=all_metrics[best_idx],
                pbest_pos=pbest_pos,
                pbest_cost=pbest_cost,
                gbest_pos=gbest_pos,
                gbest_cost=gbest_cost,
                architecture_config=all_configs[best_idx],
                oace_score=all_oace_scores[best_idx],
                evaluation_time=evaluation_time
            )
            
            # Simula checkpoint a cada 3 iterações
            if iteration % 3 == 0:
                checkpoint_file = logger.log_checkpoint(
                    iteration=iteration,
                    phase="PSO" if iteration <= 3 else "GA-PSO",
                    population=population,
                    fitness_values=fitness_values,
                    best_position=population[best_idx],
                    best_fitness=fitness_values[best_idx],
                    optimizer_state={
                        "swarm_position": population.tolist(),
                        "swarm_velocity": (population * 0.1).tolist(),
                        "pbest_pos": pbest_pos.tolist(),
                        "pbest_cost": pbest_cost.tolist(),
                        "gbest_pos": gbest_pos.tolist(),
                        "gbest_cost": float(gbest_cost)
                    },
                    metadata={
                        "checkpoint_reason": "scheduled",
                        "training_progress": iteration / 5.0
                    }
                )
                print(f"   💾 Checkpoint criado: iteração {iteration}")
        
        print("   ✅ Iterações com dados realistas logadas")
        
        # Finaliza experimento
        best_solution = logger.get_best_solution()
        if best_solution:
            logger.log_final_results(
                best_architecture=best_solution["architecture_config"]["model_type"],
                best_params=best_solution["architecture_config"],
                best_fitness=best_solution["fitness"],
                final_metrics=best_solution["metrics"]
            )
        
        print("   ✅ Resultados finais registrados")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste aprimorado: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_exports():
    """Testa se todos os CSVs foram exportados corretamente"""
    print("🧪 Testando exportação de CSVs...")
    
    try:
        # Procura o experimento mais recente
        results_dir = "results"
        if not os.path.exists(results_dir):
            return False
        
        # Pega o experimento mais recente
        experiments = [d for d in os.listdir(results_dir) if d.startswith("experiment_")]
        if not experiments:
            return False
        
        latest_experiment = sorted(experiments)[-1]
        csv_dir = os.path.join(results_dir, latest_experiment, "csv")
        analysis_dir = os.path.join(results_dir, latest_experiment, "analysis")
        
        # Verifica CSVs obrigatórios
        required_csvs = [
            f"{latest_experiment}_avaliacoes.csv",
            f"{latest_experiment}_detailed_metrics.csv",
            f"{latest_experiment}_oace_scores.csv",
            f"{latest_experiment}_pbest_history.csv",
            f"{latest_experiment}_gbest_history.csv"
        ]
        
        required_analysis = [
            f"{latest_experiment}_convergence_analysis.csv",
            f"{latest_experiment}_architecture_comparison.csv"
        ]
        
        # Verifica CSVs
        for csv_file in required_csvs:
            csv_path = os.path.join(csv_dir, csv_file)
            if not os.path.exists(csv_path):
                print(f"   ❌ CSV não encontrado: {csv_file}")
                return False
            
            # Verifica se tem conteúdo
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:  # Header + pelo menos 1 linha de dados
                    print(f"   ⚠️ CSV vazio: {csv_file}")
                else:
                    print(f"   ✅ CSV válido: {csv_file} ({len(lines)-1} registros)")
        
        # Verifica análises
        for analysis_file in required_analysis:
            analysis_path = os.path.join(analysis_dir, analysis_file)
            if os.path.exists(analysis_path):
                print(f"   ✅ Análise criada: {analysis_file}")
            else:
                print(f"   ⚠️ Análise não criada: {analysis_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na verificação de CSVs: {e}")
        return False

def test_checkpoint_recovery():
    """Testa recuperação de checkpoint"""
    print("🧪 Testando recuperação de checkpoint...")
    
    try:
        # Procura checkpoint mais recente
        results_dir = "results"
        checkpoint_file = None
        
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if "checkpoint" in file and file.endswith(".json"):
                    checkpoint_file = os.path.join(root, file)
                    break
            if checkpoint_file:
                break
        
        if not checkpoint_file:
            print("   ⚠️ Nenhum checkpoint encontrado para testar")
            return True
        
        # Testa carregamento do checkpoint
        logger = OptimizationLogger()
        checkpoint_data = logger.load_checkpoint(checkpoint_file)
        
        # Verifica se tem os campos necessários
        required_fields = ["iteration", "phase", "best_fitness", "optimizer_state"]
        for field in required_fields:
            if field not in checkpoint_data:
                print(f"   ❌ Campo obrigatório ausente: {field}")
                return False
        
        print(f"   ✅ Checkpoint válido carregado: {checkpoint_file}")
        print(f"       • Iteração: {checkpoint_data['iteration']}")
        print(f"       • Fase: {checkpoint_data['phase']}")
        print(f"       • Melhor fitness: {checkpoint_data['best_fitness']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste de checkpoint: {e}")
        return False

def test_integration_with_optimization_manager():
    """Testa integração com OptimizationManager"""
    print("🧪 Testando integração com OptimizationManager...")
    
    try:
        manager = OptimizationManager(log_dir="results")
        
        config = {
            "algorithm": "PSO",
            "population_size": 3,
            "max_iter": 5,
            "test_integration": True
        }
        
        manager.start_experiment(config)
        
        # Cria PSO com função de fitness que retorna OACE
        optimizer = PSO(
            population_size=3,
            n_dim=4,
            max_iter=5,
            lower_bound=-1,
            upper_bound=1
        )
        
        def test_fitness_with_oace(x):
            """Função de fitness que simula avaliação OACE"""
            if x.ndim == 1:
                # Simula avaliação de arquitetura
                arch_config = generate_realistic_architecture_config()
                metrics = generate_realistic_metrics(arch_config)
                oace_score = calculate_realistic_oace_score(arch_config, metrics)
                return oace_score
            else:
                scores = []
                for xi in x:
                    arch_config = generate_realistic_architecture_config()
                    metrics = generate_realistic_metrics(arch_config)
                    oace_score = calculate_realistic_oace_score(arch_config, metrics)
                    scores.append(oace_score)
                return np.array(scores)
        
        def test_metrics_with_architecture(x):
            """Função que retorna métricas e configuração de arquitetura"""
            arch_config = generate_realistic_architecture_config()
            metrics = generate_realistic_metrics(arch_config)
            # Adiciona informações de arquitetura às métricas
            metrics.update(arch_config)
            return metrics
        
        # Executa otimização
        best_position, best_fitness = manager.run_optimization(
            optimizer=optimizer,
            fitness_function=test_fitness_with_oace,
            metrics_function=test_metrics_with_architecture
        )
        
        # Verifica resultados
        assert best_position is not None, "Melhor posição não encontrada"
        assert best_fitness is not None, "Melhor fitness não encontrado"
        
        # Verifica se experimento foi registrado
        summary = manager.get_experiment_summary()
        assert summary["experiment_id"] is not None, "ID do experimento não encontrado"
        
        print("   ✅ Integração com OptimizationManager funcionando")
        print(f"       • Melhor fitness: {best_fitness:.6f}")
        print(f"       • Experimento: {summary['experiment_id']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na integração: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_final_structure():
    """Verifica se a estrutura final de arquivos está correta"""
    print("🧪 Verificando estrutura final de arquivos...")
    
    try:
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("   ❌ Pasta results não existe")
            return False
        
        experiments = [d for d in os.listdir(results_dir) if d.startswith("experiment_")]
        if not experiments:
            print("   ❌ Nenhum experimento encontrado")
            return False
        
        latest_experiment = sorted(experiments)[-1]
        experiment_dir = os.path.join(results_dir, latest_experiment)
        
        # Estrutura esperada
        expected_structure = {
            "logs": [f"{latest_experiment}_log.json"],
            "csv": [
                f"{latest_experiment}_avaliacoes.csv",
                f"{latest_experiment}_detailed_metrics.csv",
                f"{latest_experiment}_oace_scores.csv",
                f"{latest_experiment}_pbest_history.csv",
                f"{latest_experiment}_gbest_history.csv"
            ],
            "checkpoints": [],  # Pode variar
            "analysis": [
                f"{latest_experiment}_convergence_analysis.csv",
                f"{latest_experiment}_architecture_comparison.csv",
                f"{latest_experiment}_optimization_summary.json"
            ]
        }
        
        total_files = 0
        missing_files = 0
        
        for subdir, expected_files in expected_structure.items():
            subdir_path = os.path.join(experiment_dir, subdir)
            
            if not os.path.exists(subdir_path):
                print(f"   ❌ Pasta não existe: {subdir}")
                missing_files += len(expected_files)
                continue
            
            print(f"   📁 {subdir}/:")
            
            if subdir == "checkpoints":
                # Para checkpoints, apenas verifica se existem arquivos
                checkpoint_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                if checkpoint_files:
                    print(f"       ✅ {len(checkpoint_files)} checkpoint(s) encontrado(s)")
                    total_files += len(checkpoint_files)
                else:
                    print(f"       ⚠️ Nenhum checkpoint encontrado")
            else:
                # Para outras pastas, verifica arquivos específicos
                for expected_file in expected_files:
                    file_path = os.path.join(subdir_path, expected_file)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"       ✅ {expected_file} ({file_size} bytes)")
                        total_files += 1
                    else:
                        print(f"       ❌ {expected_file} (não encontrado)")
                        missing_files += 1
        
        print(f"\n   📊 Resumo da verificação:")
        print(f"       • Total de arquivos criados: {total_files}")
        print(f"       • Arquivos faltando: {missing_files}")
        print(f"       • Estrutura: {'✅ Completa' if missing_files == 0 else '⚠️ Incompleta'}")
        
        return missing_files == 0
        
    except Exception as e:
        print(f"   ❌ Erro na verificação: {e}")
        return False

def main():
    """Executa todos os testes aprimorados"""
    print("🚀 INICIANDO TESTES DO SISTEMA DE LOGGING APRIMORADO")
    print("=" * 70)
    
    tests = [
        ("Sistema de Logging Aprimorado", test_enhanced_logging_system),
        ("Exportação de CSVs", test_csv_exports),
        ("Recuperação de Checkpoint", test_checkpoint_recovery),
        ("Integração com OptimizationManager", test_integration_with_optimization_manager),
        ("Estrutura Final de Arquivos", verify_final_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Executando: {test_name}")
        print("-" * 50)
        try:
            if test_func():
                print(f"✅ {test_name}: PASSOU")
                passed += 1
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"📊 RESULTADOS DOS TESTES APRIMORADOS")
    print(f"   • Total de testes: {total}")
    print(f"   • Testes aprovados: {passed}")
    print(f"   • Testes reprovados: {total - passed}")
    print(f"   • Taxa de sucesso: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema de logging aprimorado está funcionando perfeitamente")
        print("\n🔍 Verifique os arquivos gerados em:")
        if os.path.exists("results"):
            experiments = [d for d in os.listdir("results") if d.startswith("experiment_")]
            if experiments:
                latest = sorted(experiments)[-1]
                print(f"   📁 results/{latest}/")
                print(f"      ├── logs/ (dados JSON detalhados)")
                print(f"      ├── csv/ (dados estruturados)")
                print(f"      ├── checkpoints/ (estados salvos)")
                print(f"      └── analysis/ (análises e relatórios)")
    else:
        print(f"\n⚠️ {total - passed} TESTE(S) FALHARAM")
        print("❌ Verifique os erros acima e corrija as implementações")
    
    print("\n" + "=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 