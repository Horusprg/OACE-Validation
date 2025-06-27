import json
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import csv
from optimizers.pso import PSO
import pandas as pd

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
        
        # Cria o diretório de logs 
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
                     metrics: Dict[str, float] = None,
                     pbest_pos=None,
                     pbest_cost=None,
                     gbest_pos=None,
                     gbest_cost=None):
        """
        Registra uma iteração do algoritmo, incluindo histórico de pbest/gbest se fornecido.
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
        if pbest_pos is not None:
            iteration_data["pbest_pos"] = pbest_pos.tolist() if isinstance(pbest_pos, np.ndarray) else pbest_pos
        if pbest_cost is not None:
            iteration_data["pbest_cost"] = pbest_cost.tolist() if isinstance(pbest_cost, np.ndarray) else pbest_cost
        if gbest_pos is not None:
            iteration_data["gbest_pos"] = gbest_pos.tolist() if isinstance(gbest_pos, np.ndarray) else gbest_pos
        if gbest_cost is not None:
            iteration_data["gbest_cost"] = float(gbest_cost)
        self.log_data["iterations"].append(iteration_data)
        
        # Registra como melhor solução 
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

    def export_to_csv(self, filename=None):
        """
        Exporta todas as arquiteturas avaliadas, métricas e scores OACE para um CSV legível.
        """
        if filename is None:
            filename = os.path.join(self.log_dir, self.current_experiment, "avaliacoes_arquiteturas.csv")
        fieldnames = [
            "iteration", "phase", "best_fitness", "best_position", "metrics"
        ]
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for it in self.log_data["iterations"]:
                row = {
                    "iteration": it.get("iteration"),
                    "phase": it.get("phase"),
                    "best_fitness": it.get("best_fitness"),
                    "best_position": it.get("best_position"),
                    "metrics": json.dumps(it.get("metrics")) if it.get("metrics") else ""
                }
                writer.writerow(row)

# Caminho do checkpoint salvo automaticamente
checkpoint_path = "results/experiment_20240610_153000/pso_checkpoint_10.json"

pso = PSO()
pso.load_state(checkpoint_path)

pso.optimize()

csv_path = "results/experiment_20240610_153000/avaliacoes_arquiteturas.csv"
df = pd.read_csv(csv_path)
print(df.head()) 