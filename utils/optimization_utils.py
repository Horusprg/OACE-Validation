"""
Utilitários para otimização e logging aprimorado do sistema AFSA-GA-PSO.
Fornece funções auxiliares para gerenciar experimentos, checkpoints e análise de resultados.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .optimization_logger import OptimizationLogger
from optimizers.pso import PSO

class OptimizationManager:
    """
    Gerenciador de otimização que facilita o uso do sistema de logging aprimorado.
    Fornece métodos para iniciar, pausar, retomar e analisar experimentos de otimização.
    """
    
    def __init__(self, log_dir: str = "results"):
        """
        Inicializa o gerenciador de otimização.
        
        Args:
            log_dir (str): Diretório onde os logs serão salvos
        """
        self.log_dir = log_dir
        self.logger = OptimizationLogger(log_dir)
        self.current_experiment = None
        self.optimizer = None
        
    def start_experiment(self, config: Dict[str, Any], optimizer: PSO = None):
        """
        Inicia um novo experimento de otimização.
        
        Args:
            config (Dict[str, Any]): Configurações do experimento
            optimizer (PSO): Instância do otimizador (opcional)
        """
        self.logger.start_experiment(config)
        self.current_experiment = self.logger.current_experiment
        
        if optimizer:
            self.optimizer = optimizer
            self.optimizer.logger = self.logger
            
        print(f"🚀 Experimento iniciado: {self.current_experiment}")
        print(f"📁 Logs salvos em: {os.path.join(self.log_dir, self.current_experiment)}")
        
    def run_optimization(self, 
                        optimizer: PSO,
                        fitness_function,
                        metrics_function=None,
                        max_iterations: int = None) -> Tuple[np.ndarray, float]:
        """
        Executa a otimização com logging completo.
        
        Args:
            optimizer (PSO): Instância do otimizador
            fitness_function: Função de fitness
            metrics_function: Função de métricas (opcional)
            max_iterations (int): Número máximo de iterações (opcional)
            
        Returns:
            Tuple[np.ndarray, float]: (melhor posição, melhor fitness)
        """
        if not self.current_experiment:
            raise ValueError("Experimento não iniciado. Chame start_experiment() primeiro.")
            
        self.optimizer = optimizer
        self.optimizer.logger = self.logger
        
        if max_iterations:
            self.optimizer.max_iter = max_iterations
            
        print(f"🔄 Iniciando otimização...")
        print(f"   • Máximo de iterações: {self.optimizer.max_iter}")
        print(f"   • Tamanho da população: {self.optimizer.population_size}")
        print(f"   • Dimensões: {self.optimizer.n_dim}")
        
        try:
            best_position, best_fitness = self.optimizer.optimize(
                fitness_function=fitness_function,
                metrics_function=metrics_function
            )
            
            print(f"✅ Otimização concluída!")
            print(f"   • Melhor posição: {best_position}")
            print(f"   • Melhor fitness: {best_fitness}")
            
            return best_position, best_fitness
            
        except Exception as e:
            print(f"❌ Erro durante a otimização: {e}")
            raise
    
    def pause_optimization(self, reason: str = "Pausa manual"):
        """
        Pausa a otimização atual e salva um checkpoint.
        
        Args:
            reason (str): Motivo da pausa
        """
        if not self.optimizer or not self.current_experiment:
            raise ValueError("Nenhuma otimização em andamento.")
            
        # Salva checkpoint manual
        checkpoint_data = {
            "iteration": len(self.logger.log_data["iterations"]),
            "phase": "MANUAL_PAUSE",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "optimizer_state": self.optimizer.get_optimizer_state()
        }
        
        checkpoint_file = os.path.join(
            self.log_dir,
            self.current_experiment,
            f"manual_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        print(f"⏸️ Otimização pausada: {checkpoint_file}")
        print(f"📝 Motivo: {reason}")
        
    def resume_from_checkpoint(self, 
                              checkpoint_file: str,
                              optimizer: PSO,
                              fitness_function,
                              metrics_function=None) -> Tuple[np.ndarray, float]:
        """
        Retoma a otimização a partir de um checkpoint.
        
        Args:
            checkpoint_file (str): Caminho do arquivo de checkpoint
            optimizer (PSO): Instância do otimizador
            fitness_function: Função de fitness
            metrics_function: Função de métricas (opcional)
            
        Returns:
            Tuple[np.ndarray, float]: (melhor posição, melhor fitness)
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_file}")
            
        print(f"🔄 Retomando otimização de: {checkpoint_file}")
        
        # Carrega o checkpoint
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        # Inicia novo experimento se necessário
        if not self.current_experiment:
            config = {
                "resumed_from_checkpoint": checkpoint_file,
                "original_iteration": checkpoint_data.get("iteration", 0),
                "original_phase": checkpoint_data.get("phase", "Unknown")
            }
            self.start_experiment(config)
            
        self.optimizer = optimizer
        self.optimizer.logger = self.logger
        
        # Retoma a otimização
        return self.optimizer.resume_optimization(
            checkpoint_file=checkpoint_file,
            fitness_function=fitness_function,
            metrics_function=metrics_function
        )
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo do experimento atual.
        
        Returns:
            Dict[str, Any]: Resumo do experimento
        """
        if not self.current_experiment:
            return {"status": "no_experiment"}
            
        return {
            "experiment_id": self.current_experiment,
            "log_dir": os.path.join(self.log_dir, self.current_experiment),
            "progress": self.logger.get_optimization_progress(),
            "best_solution": self.logger.get_best_solution(),
            "files_available": self._get_available_files()
        }
    
    def _get_available_files(self) -> List[str]:
        """Retorna lista de arquivos disponíveis no experimento."""
        if not self.current_experiment:
            return []
            
        experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        if not os.path.exists(experiment_dir):
            return []
            
        files = []
        for file in os.listdir(experiment_dir):
            if file.endswith(('.json', '.csv')):
                files.append(file)
        return sorted(files)
    
    def analyze_results(self, experiment_id: str = None) -> Dict[str, Any]:
        """
        Analisa os resultados de um experimento.
        
        Args:
            experiment_id (str): ID do experimento (se None, usa o atual)
            
        Returns:
            Dict[str, Any]: Análise dos resultados
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
            
        if not experiment_id:
            raise ValueError("Nenhum experimento especificado.")
            
        experiment_dir = os.path.join(self.log_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"Experimento não encontrado: {experiment_id}")
            
        # Carrega dados do experimento
        log_file = os.path.join(experiment_dir, "optimization_log.json")
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Log não encontrado: {log_file}")
            
        with open(log_file, 'r') as f:
            log_data = json.load(f)
            
        # Análise dos resultados
        analysis = {
            "experiment_id": experiment_id,
            "total_iterations": len(log_data["iterations"]),
            "total_evaluations": log_data["optimization_summary"]["total_evaluations"],
            "best_score": log_data["optimization_summary"]["best_score"],
            "convergence_iteration": log_data["optimization_summary"]["convergence_iteration"],
            "phases_executed": list(set([it["phase"] for it in log_data["iterations"]])),
            "checkpoints_created": len(log_data["checkpoints"]),
            "best_solutions_found": len(log_data["best_solutions"])
        }
        
        # Análise de convergência
        if log_data["iterations"]:
            fitness_history = [it["best_fitness"] for it in log_data["iterations"]]
            analysis["fitness_progression"] = {
                "initial_fitness": fitness_history[0],
                "final_fitness": fitness_history[-1],
                "improvement": fitness_history[-1] - fitness_history[0],
                "max_fitness": max(fitness_history),
                "min_fitness": min(fitness_history)
            }
            
        return analysis
    
    def export_experiment_data(self, experiment_id: str = None, format: str = "all"):
        """
        Exporta dados do experimento em diferentes formatos.
        
        Args:
            experiment_id (str): ID do experimento (se None, usa o atual)
            format (str): Formato de exportação ("all", "csv", "json", "summary")
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
            
        if not experiment_id:
            raise ValueError("Nenhum experimento especificado.")
            
        experiment_dir = os.path.join(self.log_dir, experiment_id)
        
        if format in ["all", "summary"]:
            # Exporta resumo em JSON
            summary_file = os.path.join(experiment_dir, "experiment_summary.json")
            analysis = self.analyze_results(experiment_id)
            
            with open(summary_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            print(f"📊 Resumo exportado: {summary_file}")
            
        if format in ["all", "csv"]:
            # Os CSVs já são criados automaticamente pelo logger
            print(f"📋 CSVs disponíveis em: {experiment_dir}")
            
        if format in ["all", "json"]:
            # O log JSON principal já é criado automaticamente
            print(f"📄 Log JSON disponível em: {experiment_dir}/optimization_log.json")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Lista todos os experimentos disponíveis.
        
        Returns:
            List[Dict[str, Any]]: Lista de experimentos com informações básicas
        """
        experiments = []
        
        if not os.path.exists(self.log_dir):
            return experiments
            
        for item in os.listdir(self.log_dir):
            if item.startswith("experiment_"):
                experiment_dir = os.path.join(self.log_dir, item)
                if os.path.isdir(experiment_dir):
                    # Tenta carregar informações básicas
                    log_file = os.path.join(experiment_dir, "optimization_log.json")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                log_data = json.load(f)
                                
                            experiments.append({
                                "id": item,
                                "timestamp": log_data["experiment_info"]["timestamp"],
                                "total_iterations": len(log_data["iterations"]),
                                "best_score": log_data["optimization_summary"]["best_score"],
                                "status": "completed" if log_data["final_results"] else "in_progress"
                            })
                        except:
                            experiments.append({
                                "id": item,
                                "timestamp": "unknown",
                                "total_iterations": 0,
                                "best_score": 0.0,
                                "status": "corrupted"
                            })
                    else:
                        experiments.append({
                            "id": item,
                            "timestamp": "unknown",
                            "total_iterations": 0,
                            "best_score": 0.0,
                            "status": "no_log"
                        })
                        
        return sorted(experiments, key=lambda x: x["timestamp"], reverse=True)

def create_optimization_experiment(config: Dict[str, Any], 
                                 log_dir: str = "results") -> OptimizationManager:
    """
    Função utilitária para criar rapidamente um experimento de otimização.
    
    Args:
        config (Dict[str, Any]): Configurações do experimento
        log_dir (str): Diretório de logs
        
    Returns:
        OptimizationManager: Gerenciador de otimização configurado
    """
    manager = OptimizationManager(log_dir)
    manager.start_experiment(config)
    return manager

def load_experiment_results(experiment_id: str, log_dir: str = "results") -> Dict[str, Any]:
    """
    Carrega resultados de um experimento específico.
    
    Args:
        experiment_id (str): ID do experimento
        log_dir (str): Diretório de logs
        
    Returns:
        Dict[str, Any]: Dados do experimento
    """
    log_file = os.path.join(log_dir, experiment_id, "optimization_log.json")
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Experimento não encontrado: {experiment_id}")
        
    with open(log_file, 'r') as f:
        return json.load(f)

# Exemplo de uso:
if __name__ == "__main__":
    # Criar gerenciador
    manager = OptimizationManager()
    
    # Configuração do experimento
    config = {
        "population_size": 10,
        "max_iter": 20,
        "lambda_param": 0.5,
        "description": "Teste do sistema de logging aprimorado"
    }
    
    # Iniciar experimento
    manager.start_experiment(config)
    
    # Criar otimizador
    optimizer = PSO(
        population_size=10,
        n_dim=5,
        max_iter=20,
        lower_bound=-10,
        upper_bound=10
    )
    
    # Função de fitness simples para teste
    def test_fitness(x):
        return -np.sum(x**2, axis=1)
    
    # Executar otimização
    best_pos, best_fitness = manager.run_optimization(
        optimizer=optimizer,
        fitness_function=test_fitness
    )
    
    # Analisar resultados
    summary = manager.get_experiment_summary()
    print(f"📊 Resumo do experimento: {summary}")
    
    # Listar todos os experimentos
    experiments = manager.list_experiments()
    print(f"📋 Experimentos disponíveis: {len(experiments)}")
    for exp in experiments:
        print(f"   • {exp['id']}: {exp['status']} (score: {exp['best_score']:.4f})") 