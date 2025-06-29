"""
Exemplo prático demonstrando o sistema de logging aprimorado para PSO.
Este script mostra como usar todas as funcionalidades implementadas:
- Logging completo de iterações
- Histórico de pbest/gbest
- Checkpoints automáticos e manuais
- Retomada de otimização
- Análise de resultados
- Exportação de dados
"""

import sys
import os
import numpy as np
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.pso import PSO
from utils.optimization_utils import OptimizationManager, create_optimization_experiment
from utils.optimization_logger import OptimizationLogger

def demo_basic_logging():
    """
    Demonstração básica do sistema de logging.
    """
    print("=" * 60)
    print("DEMONSTRAÇÃO 1: LOGGING BÁSICO")
    print("=" * 60)
    
    # 1. Criar gerenciador de otimização
    manager = OptimizationManager(log_dir="results/demo")
    
    # 2. Configuração do experimento
    config = {
        "population_size": 5,
        "max_iter": 10,
        "lambda_param": 0.5,
        "description": "Demonstração básica do sistema de logging",
        "algorithm": "PSO",
        "test_type": "basic_logging"
    }
    
    # 3. Iniciar experimento
    manager.start_experiment(config)
    
    # 4. Criar otimizador PSO
    optimizer = PSO(
        population_size=5,
        n_dim=3,
        max_iter=10,
        lower_bound=-5,
        upper_bound=5
    )
    
    # 5. Função de fitness para demonstração
    def demo_fitness(x):
        """Função de fitness simples para demonstração."""
        return -np.sum(x**2, axis=1)  # Maximiza o negativo da soma dos quadrados
    
    # 6. Função de métricas simulada
    def demo_metrics(x):
        """Simula métricas de uma arquitetura de rede neural."""
        # Simula métricas de assertividade
        assertiveness = np.random.uniform(0.7, 0.95)
        
        # Simula métricas de custo
        total_params = np.random.randint(100000, 1000000)
        inference_time = np.random.uniform(0.01, 0.1)
        memory_used = np.random.uniform(50, 500)
        gflops = np.random.uniform(1, 10)
        
        return {
            "top1_acc": assertiveness,
            "top5_acc": assertiveness + 0.05,
            "precision_macro": assertiveness - 0.02,
            "recall_macro": assertiveness - 0.01,
            "f1_macro": assertiveness - 0.015,
            "total_params": total_params,
            "avg_inference_time": inference_time,
            "memory_used_mb": memory_used,
            "gflops": gflops
        }
    
    # 7. Executar otimização
    print("\n🔄 Executando otimização...")
    best_position, best_fitness = manager.run_optimization(
        optimizer=optimizer,
        fitness_function=demo_fitness,
        metrics_function=demo_metrics
    )
    
    # 8. Analisar resultados
    print("\n📊 Analisando resultados...")
    summary = manager.get_experiment_summary()
    print(f"   • ID do experimento: {summary['experiment_id']}")
    print(f"   • Progresso: {summary['progress']}")
    print(f"   • Melhor solução: {summary['best_solution']}")
    print(f"   • Arquivos disponíveis: {summary['files_available']}")
    
    # 9. Exportar dados
    print("\n📁 Exportando dados...")
    manager.export_experiment_data(format="all")
    
    return manager.current_experiment

def demo_checkpoint_resume():
    """
    Demonstração de checkpoint e retomada de otimização.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO 2: CHECKPOINT E RETOMADA")
    print("=" * 60)
    
    # 1. Criar gerenciador
    manager = OptimizationManager(log_dir="results/demo")
    
    # 2. Configuração
    config = {
        "population_size": 3,
        "max_iter": 15,
        "lambda_param": 0.5,
        "description": "Demonstração de checkpoint e retomada",
        "algorithm": "PSO",
        "test_type": "checkpoint_resume"
    }
    
    # 3. Iniciar experimento
    manager.start_experiment(config)
    
    # 4. Criar otimizador
    optimizer = PSO(
        population_size=3,
        n_dim=2,
        max_iter=15,
        lower_bound=-3,
        upper_bound=3
    )
    
    # 5. Função de fitness
    def demo_fitness(x):
        return -np.sum(x**2, axis=1)
    
    # 6. Função de métricas
    def demo_metrics(x):
        return {
            "top1_acc": np.random.uniform(0.8, 0.95),
            "total_params": np.random.randint(50000, 500000),
            "avg_inference_time": np.random.uniform(0.005, 0.05),
            "memory_used_mb": np.random.uniform(25, 250),
            "gflops": np.random.uniform(0.5, 5)
        }
    
    # 7. Executar otimização (será interrompida)
    print("\n🔄 Executando otimização (será interrompida)...")
    
    # Simula interrupção após algumas iterações
    try:
        best_position, best_fitness = manager.run_optimization(
            optimizer=optimizer,
            fitness_function=demo_fitness,
            metrics_function=demo_metrics
        )
    except KeyboardInterrupt:
        print("\n⏸️ Otimização interrompida pelo usuário")
        
        # Salva checkpoint manual
        manager.pause_optimization("Interrupção manual para demonstração")
        
        # Simula retomada
        print("\n🔄 Retomando otimização...")
        
        # Cria novo otimizador
        optimizer2 = PSO(
            population_size=3,
            n_dim=2,
            max_iter=15,
            lower_bound=-3,
            upper_bound=3
        )
        
        # Encontra o último checkpoint
        experiment_dir = os.path.join("results/demo", manager.current_experiment)
        checkpoint_files = [f for f in os.listdir(experiment_dir) if f.startswith("checkpoint")]
        
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint_path = os.path.join(experiment_dir, latest_checkpoint)
            
            # Retoma a otimização
            best_position, best_fitness = manager.resume_from_checkpoint(
                checkpoint_file=checkpoint_path,
                optimizer=optimizer2,
                fitness_function=demo_fitness,
                metrics_function=demo_metrics
            )
            
            print(f"✅ Otimização retomada e concluída!")
            print(f"   • Melhor posição: {best_position}")
            print(f"   • Melhor fitness: {best_fitness}")
    
    return manager.current_experiment

def demo_analysis_and_export():
    """
    Demonstração de análise e exportação de resultados.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO 3: ANÁLISE E EXPORTAÇÃO")
    print("=" * 60)
    
    # 1. Criar gerenciador
    manager = OptimizationManager(log_dir="results/demo")
    
    # 2. Listar experimentos existentes
    print("\n📋 Experimentos disponíveis:")
    experiments = manager.list_experiments()
    for exp in experiments:
        print(f"   • {exp['id']}: {exp['status']} (score: {exp['best_score']:.4f})")
    
    # 3. Analisar um experimento específico (se existir)
    if experiments:
        experiment_id = experiments[0]['id']
        print(f"\n📊 Analisando experimento: {experiment_id}")
        
        analysis = manager.analyze_results(experiment_id)
        print(f"   • Total de iterações: {analysis['total_iterations']}")
        print(f"   • Total de avaliações: {analysis['total_evaluations']}")
        print(f"   • Melhor score: {analysis['best_score']:.4f}")
        print(f"   • Fases executadas: {analysis['phases_executed']}")
        print(f"   • Checkpoints criados: {analysis['checkpoints_created']}")
        
        if 'fitness_progression' in analysis:
            prog = analysis['fitness_progression']
            print(f"   • Progressão do fitness:")
            print(f"     - Inicial: {prog['initial_fitness']:.4f}")
            print(f"     - Final: {prog['final_fitness']:.4f}")
            print(f"     - Melhoria: {prog['improvement']:.4f}")
            print(f"     - Máximo: {prog['max_fitness']:.4f}")
            print(f"     - Mínimo: {prog['min_fitness']:.4f}")
        
        # 4. Exportar dados
        print(f"\n📁 Exportando dados do experimento...")
        manager.export_experiment_data(experiment_id, format="all")
        
        # 5. Carregar dados brutos
        print(f"\n📄 Carregando dados brutos...")
        raw_data = manager.logger.load_checkpoint(
            os.path.join("results/demo", experiment_id, "optimization_log.json")
        )
        print(f"   • Dados carregados com sucesso")
        print(f"   • Estrutura: {list(raw_data.keys())}")

def demo_advanced_logging():
    """
    Demonstração de logging avançado com múltiplas fases.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO 4: LOGGING AVANÇADO")
    print("=" * 60)
    
    # 1. Criar logger diretamente
    logger = OptimizationLogger(log_dir="results/demo")
    
    # 2. Configuração
    config = {
        "population_size": 4,
        "max_iter": 8,
        "lambda_param": 0.5,
        "description": "Demonstração de logging avançado",
        "algorithm": "PSO",
        "test_type": "advanced_logging"
    }
    
    # 3. Iniciar experimento
    logger.start_experiment(config)
    
    # 4. Simular múltiplas fases de otimização
    phases = ["AFSA", "PSO", "GA"]
    
    for phase_idx, phase in enumerate(phases):
        print(f"\n🔄 Executando fase: {phase}")
        
        for iteration in range(1, 4):  # 3 iterações por fase
            # Simula população
            population = np.random.rand(4, 3)
            fitness_values = np.random.rand(4)
            best_position = population[np.argmax(fitness_values)]
            best_fitness = np.max(fitness_values)
            
            # Simula pbest/gbest
            pbest_pos = population.copy()
            pbest_cost = fitness_values.copy()
            gbest_pos = best_position.copy()
            gbest_cost = best_fitness
            
            # Simula métricas
            metrics = {
                "top1_acc": np.random.uniform(0.8, 0.95),
                "top5_acc": np.random.uniform(0.85, 0.98),
                "precision_macro": np.random.uniform(0.75, 0.9),
                "recall_macro": np.random.uniform(0.75, 0.9),
                "f1_macro": np.random.uniform(0.75, 0.9),
                "total_params": np.random.randint(100000, 1000000),
                "avg_inference_time": np.random.uniform(0.01, 0.1),
                "memory_used_mb": np.random.uniform(50, 500),
                "gflops": np.random.uniform(1, 10)
            }
            
            # Simula configuração de arquitetura
            architecture_config = {
                "model_type": f"TestModel_{phase}",
                "layers": np.random.randint(3, 8),
                "neurons": np.random.randint(64, 512),
                "dropout": np.random.uniform(0.1, 0.5)
            }
            
            # Log da iteração
            logger.log_iteration(
                iteration=iteration + phase_idx * 3,
                phase=phase,
                population=population,
                fitness_values=fitness_values,
                best_position=best_position,
                best_fitness=best_fitness,
                metrics=metrics,
                pbest_pos=pbest_pos,
                pbest_cost=pbest_cost,
                gbest_pos=gbest_pos,
                gbest_cost=gbest_cost,
                architecture_config=architecture_config
            )
            
            # Checkpoint a cada 3 iterações
            if iteration % 3 == 0:
                logger.log_checkpoint(
                    iteration=iteration + phase_idx * 3,
                    phase=phase,
                    population=population,
                    fitness_values=fitness_values,
                    best_position=best_position,
                    best_fitness=best_fitness
                )
    
    # 5. Finalizar experimento
    logger.log_final_results(
        best_architecture="TestModel_PSO",
        best_params={"layers": 5, "neurons": 256, "dropout": 0.3},
        best_fitness=0.92,
        final_metrics={"top1_acc": 0.92, "total_params": 500000}
    )
    
    print(f"\n✅ Logging avançado concluído!")
    print(f"   • Experimento: {logger.current_experiment}")
    print(f"   • Total de iterações: {len(logger.log_data['iterations'])}")
    print(f"   • Fases executadas: {list(set([it['phase'] for it in logger.log_data['iterations']]))}")

def main():
    """
    Função principal que executa todas as demonstrações.
    """
    print("🚀 INICIANDO DEMONSTRAÇÕES DO SISTEMA DE LOGGING APRIMORADO")
    print("=" * 80)
    
    try:
        # Demonstração 1: Logging básico
        exp1 = demo_basic_logging()
        
        # Demonstração 2: Checkpoint e retomada
        exp2 = demo_checkpoint_resume()
        
        # Demonstração 3: Análise e exportação
        demo_analysis_and_export()
        
        # Demonstração 4: Logging avançado
        demo_advanced_logging()
        
        print("\n" + "=" * 80)
        print("✅ TODAS AS DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO!")
        print("=" * 80)
        print("\n📁 Arquivos gerados:")
        print("   • results/demo/ - Diretório com todos os experimentos")
        print("   • optimization_log.json - Log completo de cada experimento")
        print("   • avaliacoes_arquiteturas.csv - Resumo em CSV")
        print("   • pbest_history.csv - Histórico de pbest")
        print("   • gbest_history.csv - Histórico de gbest")
        print("   • detailed_metrics.csv - Métricas detalhadas")
        print("   • optimization_summary.json - Resumo da otimização")
        print("   • checkpoint_*.json - Checkpoints automáticos e manuais")
        
        print("\n🔧 Funcionalidades demonstradas:")
        print("   ✅ Logging completo de iterações")
        print("   ✅ Histórico de pbest e gbest")
        print("   ✅ Checkpoints automáticos e manuais")
        print("   ✅ Retomada de otimização")
        print("   ✅ Análise de resultados")
        print("   ✅ Exportação em múltiplos formatos")
        print("   ✅ Gerenciamento de experimentos")
        
    except Exception as e:
        print(f"\n❌ Erro durante as demonstrações: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 