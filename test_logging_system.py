#!/usr/bin/env python3
"""
Script de teste para verificar o sistema de logging aprimorado.
Executa um teste rápido para validar todas as funcionalidades implementadas.
"""

import sys
import os
import numpy as np
import json

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.optimization_logger import OptimizationLogger
from utils.optimization_utils import OptimizationManager
from optimizers.pso_fixed import PSO

def test_basic_logging():
    """Testa funcionalidades básicas do sistema de logging."""
    print("🧪 Testando funcionalidades básicas...")
    
    try:
        # 1. Testar OptimizationLogger
        logger = OptimizationLogger(log_dir="results")
        
        # Configuração de teste
        config = {
            "population_size": 3,
            "max_iter": 5,
            "lambda_param": 0.5,
            "test": True
        }
        
        # Iniciar experimento
        logger.start_experiment(config)
        assert logger.current_experiment is not None, "Experimento não foi iniciado"
        print("   ✅ Experimento iniciado com sucesso")
        
        # Simular algumas iterações
        for i in range(3):
            population = np.random.rand(3, 2)
            fitness_values = np.random.rand(3)
            best_position = population[np.argmax(fitness_values)]
            best_fitness = np.max(fitness_values)
            
            metrics = {
                "top1_acc": np.random.uniform(0.8, 0.95),
                "total_params": np.random.randint(100000, 500000)
            }
            
            logger.log_iteration(
                iteration=i+1,
                phase="PSO",
                population=population,
                fitness_values=fitness_values,
                best_position=best_position,
                best_fitness=best_fitness,
                metrics=metrics,
                architecture_config={"model_type": "TestModel"}
            )
        
        print("   ✅ Iterações logadas com sucesso")
        
        # Verificar se arquivos foram criados na pasta results
        log_file = os.path.join("results", logger.current_experiment, "logs", f"{logger.current_experiment}_log.json")
        assert os.path.exists(log_file), "Arquivo de log não foi criado"
        
        csv_file = os.path.join("results", logger.current_experiment, "csv", f"{logger.current_experiment}_avaliacoes.csv")
        assert os.path.exists(csv_file), "Arquivo CSV não foi criado"
        
        print("   ✅ Arquivos de log criados com sucesso")
        
        # Verificar estrutura do log
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        assert "experiment_info" in log_data, "Informações do experimento não encontradas"
        assert "iterations" in log_data, "Iterações não encontradas"
        assert len(log_data["iterations"]) == 3, "Número incorreto de iterações"
        
        print("   ✅ Estrutura do log válida")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste básico: {e}")
        return False

def test_pso_integration():
    """Testa integração com PSO."""
    print("🧪 Testando integração com PSO...")
    
    try:
        # Criar gerenciador
        manager = OptimizationManager(log_dir="results")
        
        # Configuração
        config = {
            "population_size": 3,
            "max_iter": 3,
            "lambda_param": 0.5,
            "test": True
        }
        
        # Iniciar experimento
        manager.start_experiment(config)
        
        # Criar PSO
        optimizer = PSO(
            population_size=3,
            n_dim=2,
            max_iter=3,
            lower_bound=-1,
            upper_bound=1
        )
        
        # Função de fitness simples
        def test_fitness(x):
            return -np.sum(x**2, axis=1)
        
        # Função de métricas simulada
        def test_metrics(x):
            return {
                "top1_acc": np.random.uniform(0.8, 0.95),
                "total_params": np.random.randint(100000, 500000),
                "avg_inference_time": np.random.uniform(0.01, 0.1)
            }
        
        # Executar otimização
        best_position, best_fitness = manager.run_optimization(
            optimizer=optimizer,
            fitness_function=test_fitness,
            metrics_function=test_metrics
        )
        
        print("   ✅ Otimização PSO executada com sucesso")
        
        # Verificar resultados
        assert best_position is not None, "Melhor posição não encontrada"
        assert best_fitness is not None, "Melhor fitness não encontrado"
        
        print(f"   ✅ Melhor posição: {best_position}")
        print(f"   ✅ Melhor fitness: {best_fitness}")
        
        # Verificar arquivos gerados na pasta results
        log_file = os.path.join("results", manager.current_experiment, "logs", f"{manager.current_experiment}_log.json")
        csv_file = os.path.join("results", manager.current_experiment, "csv", f"{manager.current_experiment}_avaliacoes.csv")
        
        assert os.path.exists(log_file), f"Arquivo {log_file} não encontrado"
        assert os.path.exists(csv_file), f"Arquivo {csv_file} não encontrado"
        
        print("   ✅ Arquivos de resultado criados")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste PSO: {e}")
        return False

def test_checkpoint_functionality():
    """Testa funcionalidades de checkpoint."""
    print("🧪 Testando funcionalidades de checkpoint...")
    
    try:
        # Criar gerenciador
        manager = OptimizationManager(log_dir="results")
        
        # Configuração
        config = {
            "population_size": 2,
            "max_iter": 15,  # Aumentado para garantir checkpoints
            "lambda_param": 0.5,
            "test": True
        }
        
        # Iniciar experimento
        manager.start_experiment(config)
        
        # Criar PSO
        optimizer = PSO(
            population_size=2,
            n_dim=2,
            max_iter=15,  # Aumentado para garantir checkpoints
            lower_bound=-1,
            upper_bound=1
        )
        
        # Função de fitness
        def test_fitness(x):
            return -np.sum(x**2, axis=1)
        
        # Função de métricas
        def test_metrics(x):
            return {
                "top1_acc": np.random.uniform(0.8, 0.95),
                "total_params": np.random.randint(100000, 500000)
            }
        
        # Executar otimização
        best_position, best_fitness = manager.run_optimization(
            optimizer=optimizer,
            fitness_function=test_fitness,
            metrics_function=test_metrics
        )
        
        print("   ✅ Otimização executada com sucesso")
        
        # Verificar se checkpoints foram criados
        checkpoint_files = []
        for root, dirs, files in os.walk("results"):
            for file in files:
                if "checkpoint" in file:
                    checkpoint_files.append(os.path.join(root, file))
        
        # Verificar especificamente na pasta checkpoints do experimento
        checkpoints_dir = os.path.join("results", manager.current_experiment, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoint_files_in_dir = [f for f in os.listdir(checkpoints_dir) if "checkpoint" in f]
            print(f"   📁 Checkpoints encontrados em {checkpoints_dir}: {len(checkpoint_files_in_dir)}")
        
        assert len(checkpoint_files) > 0, "Nenhum checkpoint foi criado"
        
        print("   ✅ Checkpoints criados com sucesso")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste de checkpoint: {e}")
        return False

def test_analysis_functions():
    """Testa funções de análise de resultados."""
    print("🧪 Testando funções de análise...")
    
    try:
        # Criar gerenciador
        manager = OptimizationManager(log_dir="results")
        
        # Configuração
        config = {
            "population_size": 2,
            "max_iter": 3,
            "lambda_param": 0.5,
            "test": True
        }
        
        # Iniciar experimento
        manager.start_experiment(config)
        
        # Criar PSO
        optimizer = PSO(
            population_size=2,
            n_dim=2,
            max_iter=3,
            lower_bound=-1,
            upper_bound=1
        )
        
        # Função de fitness
        def test_fitness(x):
            return -np.sum(x**2, axis=1)
        
        # Função de métricas
        def test_metrics(x):
            return {
                "top1_acc": np.random.uniform(0.8, 0.95),
                "total_params": np.random.randint(100000, 500000)
            }
        
        # Executar otimização
        best_position, best_fitness = manager.run_optimization(
            optimizer=optimizer,
            fitness_function=test_fitness,
            metrics_function=test_metrics
        )
        
        print("   ✅ Otimização executada com sucesso")
        
        # Verificar se log foi criado
        log_file = os.path.join("results", manager.current_experiment, "logs", f"{manager.current_experiment}_log.json")
        assert os.path.exists(log_file), f"Log não encontrado: {log_file}"
        
        # Carregar e analisar log
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # Verificar estrutura básica
        assert "experiment_info" in log_data, "Informações do experimento não encontradas"
        assert "iterations" in log_data, "Iterações não encontradas"
        assert "optimization_summary" in log_data, "Resumo da otimização não encontrado"
        
        # Verificar dados das iterações
        iterations = log_data["iterations"]
        assert len(iterations) > 0, "Nenhuma iteração encontrada"
        
        # Verificar se cada iteração tem os campos necessários
        for iteration in iterations:
            assert "iteration" in iteration, "Número da iteração não encontrado"
            assert "best_fitness" in iteration, "Melhor fitness não encontrado"
            assert "best_position" in iteration, "Melhor posição não encontrada"
        
        print("   ✅ Análise de resultados executada com sucesso")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste de análise: {e}")
        return False

def main():
    """Executa todos os testes."""
    print("🚀 INICIANDO TESTES DO SISTEMA DE LOGGING APRIMORADO")
    print("=" * 60)
    
    tests = [
        ("Funcionalidades Básicas", test_basic_logging),
        ("Integração PSO", test_pso_integration),
        ("Checkpoints", test_checkpoint_functionality),
        ("Análise de Resultados", test_analysis_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Executando: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSOU")
                passed += 1
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS DOS TESTES")
    print(f"   • Total de testes: {total}")
    print(f"   • Testes aprovados: {passed}")
    print(f"   • Testes reprovados: {total - passed}")
    print(f"   • Taxa de sucesso: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema de logging está funcionando corretamente")
    else:
        print(f"\n⚠️ {total - passed} TESTE(S) FALHARAM")
        print("❌ Verifique os erros acima")
    
    print("\n" + "=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 