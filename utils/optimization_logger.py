import json
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

class OptimizationLogger:
    """
    Classe para registrar o progresso da otimização AFSA-GA-PSO.
    Armazena informações sobre cada iteração, incluindo:
    - Configurações das arquiteturas
    - Métricas de assertividade e custo
    - Scores OACE
    - Histórico de pbest e gbest
    - Checkpoints para retomada de experimentos
    """
    
    def __init__(self, log_dir: str = "results"):
        """
        Inicializa o logger.
        
        Args:
            log_dir (str): Diretório onde os logs serão salvos
        """
        self.log_dir = log_dir
        self.current_experiment = None
        self.log_data = {
            "experiment_info": {},
            "iterations": [],
            "best_solutions": [],
            "metrics_history": [],
            "checkpoints": [],
            "final_results": {}
        }
        
        # Cria o diretório de logs se não existir
        os.makedirs(log_dir, exist_ok=True)
    
    def start_experiment(self, config: Dict[str, Any]):
        """
        Inicia um novo experimento com timestamp único
        
        Args:
            config (Dict[str, Any]): Configurações do experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment = f"experiment_{timestamp}"
        
        # Cria diretório do experimento
        experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Inicializa dados do experimento
        self.log_data = {
            "experiment_info": {
                "timestamp": timestamp,
                "config": config
            },
            "iterations": [],
            "best_solutions": [],
            "metrics_history": [],
            "checkpoints": [],
            "final_results": {}
        }
        
        # Salva o log inicial
        self._save_log()
    
    def log_iteration(self, 
                     iteration: int,
                     phase: str,
                     population: np.ndarray,
                     fitness_values: np.ndarray,
                     best_position: np.ndarray,
                     best_fitness: float,
                     metrics: Dict[str, float] = None):
        """
        Registra uma iteração do algoritmo.
        
        Args:
            iteration (int): Número da iteração
            phase (str): Fase do algoritmo (AFSA, PSO, GA)
            population (np.ndarray): População atual
            fitness_values (np.ndarray): Valores de fitness da população
            best_position (np.ndarray): Melhor posição encontrada
            best_fitness (float): Melhor valor de fitness
            metrics (Dict[str, float]): Métricas da arquitetura
        """
        iteration_data = {
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "population": population.tolist() if isinstance(population, np.ndarray) else population,
            "fitness_values": fitness_values.tolist() if isinstance(fitness_values, np.ndarray) else fitness_values,
            "best_position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
            "best_fitness": float(best_fitness),
            "metrics": metrics
        }
        
        self.log_data["iterations"].append(iteration_data)
        
        # Registra como melhor solução se for a melhor até agora
        if not self.log_data["best_solutions"] or best_fitness > self.log_data["best_solutions"][-1]["fitness"]:
            self.log_data["best_solutions"].append({
                "iteration": iteration,
                "phase": phase,
                "position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                "fitness": float(best_fitness),
                "metrics": metrics
            })
        
        # Atualiza histórico de métricas
        if metrics:
            self.log_data["metrics_history"].append({
                "iteration": iteration,
                "phase": phase,
                "metrics": metrics
            })
        
        # Salva o log após cada iteração
        self._save_log()
    
    def log_checkpoint(self, 
                      iteration: int,
                      phase: str,
                      population: np.ndarray,
                      fitness_values: np.ndarray,
                      best_position: np.ndarray,
                      best_fitness: float):
        """
        Cria um checkpoint do estado atual da otimização.
        
        Args:
            iteration (int): Número da iteração
            phase (str): Fase do algoritmo
            population (np.ndarray): População atual
            fitness_values (np.ndarray): Valores de fitness
            best_position (np.ndarray): Melhor posição
            best_fitness (float): Melhor valor de fitness
        """
        checkpoint_data = {
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "population": population.tolist() if isinstance(population, np.ndarray) else population,
            "fitness_values": fitness_values.tolist() if isinstance(fitness_values, np.ndarray) else fitness_values,
            "best_position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
            "best_fitness": float(best_fitness)
        }
        
        self.log_data["checkpoints"].append(checkpoint_data)
        
        # Salva o checkpoint em um arquivo separado
        checkpoint_file = os.path.join(
            self.log_dir,
            self.current_experiment,
            f"checkpoint_{iteration}_{phase}.json"
        )
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def log_final_results(self, best_architecture, best_params, best_fitness, final_metrics):
        """Registra os resultados finais do experimento"""
        self.log_data["final_results"] = {
            "timestamp": datetime.now().isoformat(),
            "best_architecture": best_architecture,
            "best_params": best_params,
            "best_fitness": float(best_fitness),
            "final_metrics": final_metrics
        }
        
        # Salva o log final
        self._save_log()
    
    def _save_log(self):
        """Salva o log atual em um arquivo JSON."""
        if not self.current_experiment:
            return
            
        log_file = os.path.join(
            self.log_dir,
            self.current_experiment,
            "optimization_log.json"
        )
        
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Carrega um checkpoint salvo.
        
        Args:
            checkpoint_file (str): Caminho do arquivo de checkpoint
            
        Returns:
            Dict[str, Any]: Dados do checkpoint
        """
        with open(checkpoint_file, 'r') as f:
            return json.load(f) 