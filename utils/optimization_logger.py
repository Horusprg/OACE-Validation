import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
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
        Inicializa o logger de otimização.
        
        Args:
            log_dir (str): Diretório base para salvar os logs
        """
        self.log_dir = log_dir
        self.current_experiment = None
        self.log_data = {
            "experiment_info": {},
            "iterations": [],
            "best_solutions": [],
            "metrics_history": [],
            "checkpoints": [],
            "final_results": {},
            "architecture_evaluations": [],  # Novo: histórico de todas as arquiteturas avaliadas
            "pbest_history": [],             # Novo: histórico completo de pbest
            "gbest_history": [],             # Novo: histórico completo de gbest
            "optimization_summary": {
                "total_iterations": 0,
                "total_evaluations": 0,
                "start_time": None,
                "end_time": None,
                "best_fitness": float('inf'),
                "best_position": None
            }
        }
        
        # Criar estrutura de pastas
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Cria a estrutura de pastas organizada"""
        # Pasta principal de resultados
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Pasta principal do experimento (será criada quando start_experiment for chamado)
        # Por enquanto, apenas define os caminhos base
        self.experiment_dir = None
        self.logs_dir = None
        self.csv_dir = None
        self.checkpoints_dir = None
    
    def start_experiment(self, config: Dict[str, Any]):
        """
        Inicia um novo experimento de otimização.
        
        Args:
            config (Dict[str, Any]): Configuração do experimento
        """
        # Gera nome único para o experimento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment = f"experiment_{timestamp}"
        
        # Cria pasta principal do experimento
        self.experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        
        # Cria subpastas organizadas dentro do experimento
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.csv_dir = os.path.join(self.experiment_dir, "csv")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Inicializa dados do experimento
        self.log_data["experiment_info"] = {
            "experiment_name": self.current_experiment,
            "start_time": datetime.now().isoformat(),
            "config": config
        }
        
        self.log_data["optimization_summary"]["start_time"] = datetime.now().isoformat()
        self.log_data["optimization_summary"]["best_fitness"] = float('inf')  # Será convertido para string no JSON
        
        # Cria arquivo CSV inicial
        self._create_summary_csv()
        
        print(f"🚀 Experimento iniciado: {self.current_experiment}")
        print(f"📁 Pasta do experimento: {self.experiment_dir}")
        print(f"   ├── 📄 logs/ (arquivos JSON)")
        print(f"   ├── 📊 csv/ (arquivos CSV)")
        print(f"   └── 💾 checkpoints/ (checkpoints)")
    
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
                     gbest_cost=None,
                     architecture_config: Dict[str, Any] = None):
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
            "metrics": metrics,
            "architecture_config": architecture_config
        }
        
        # Adiciona dados de pbest/gbest se fornecidos
        if pbest_pos is not None:
            iteration_data["pbest_pos"] = pbest_pos.tolist() if isinstance(pbest_pos, np.ndarray) else pbest_pos
        if pbest_cost is not None:
            iteration_data["pbest_cost"] = pbest_cost.tolist() if isinstance(pbest_cost, np.ndarray) else pbest_cost
        if gbest_pos is not None:
            iteration_data["gbest_pos"] = gbest_pos.tolist() if isinstance(gbest_pos, np.ndarray) else gbest_pos
        if gbest_cost is not None:
            iteration_data["gbest_cost"] = float(gbest_cost)
            
        self.log_data["iterations"].append(iteration_data)
        
        # Registra como melhor solução se for melhor que a anterior
        if not self.log_data["best_solutions"] or best_fitness > self.log_data["best_solutions"][-1]["fitness"]:
            self.log_data["best_solutions"].append({
                "iteration": iteration,
                "phase": phase,
                "position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                "fitness": float(best_fitness),
                "metrics": metrics,
                "architecture_config": architecture_config
            })
            
            # Atualiza o resumo da otimização
            self.log_data["optimization_summary"]["best_score"] = float(best_fitness)
            self.log_data["optimization_summary"]["best_architecture"] = architecture_config
            self.log_data["optimization_summary"]["convergence_iteration"] = iteration
        
        # Atualiza histórico de métricas
        if metrics:
            self.log_data["metrics_history"].append({
                "iteration": iteration,
                "phase": phase,
                "metrics": metrics,
                "architecture_config": architecture_config
            })
        
        # Registra avaliação da arquitetura
        if architecture_config and metrics:
            self.log_data["architecture_evaluations"].append({
                "iteration": iteration,
                "phase": phase,
                "architecture_config": architecture_config,
                "metrics": metrics,
                "fitness": float(best_fitness),
                "position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position
            })
        
        # Registra histórico de pbest/gbest
        if pbest_pos is not None and pbest_cost is not None:
            self.log_data["pbest_history"].append({
                "iteration": iteration,
                "phase": phase,
                "pbest_positions": pbest_pos.tolist() if isinstance(pbest_pos, np.ndarray) else pbest_pos,
                "pbest_costs": pbest_cost.tolist() if isinstance(pbest_cost, np.ndarray) else pbest_cost
            })
        
        if gbest_pos is not None and gbest_cost is not None:
            self.log_data["gbest_history"].append({
                "iteration": iteration,
                "phase": phase,
                "gbest_position": gbest_pos.tolist() if isinstance(gbest_pos, np.ndarray) else gbest_pos,
                "gbest_cost": float(gbest_cost)
            })
        
        # Atualiza contador de avaliações
        self.log_data["optimization_summary"]["total_evaluations"] += 1
        
        # Salva o log após cada iteração
        self._save_log()
        
        # Atualiza o CSV de resumo
        self._update_summary_csv(iteration, phase, best_fitness, metrics, architecture_config)
    
    def log_checkpoint(self, 
                      iteration: int,
                      phase: str,
                      population: np.ndarray,
                      fitness_values: np.ndarray,
                      best_position: np.ndarray,
                      best_fitness: float,
                      optimizer_state: Dict[str, Any] = None):
        """
        Cria um checkpoint do estado atual da otimização.
        
        Args:
            iteration (int): Número da iteração
            phase (str): Fase do algoritmo
            population (np.ndarray): População atual
            fitness_values (np.ndarray): Valores de fitness
            best_position (np.ndarray): Melhor posição
            best_fitness (float): Melhor valor de fitness
            optimizer_state (Dict[str, Any]): Estado interno do otimizador (opcional)
        """
        checkpoint_data = {
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "population": population.tolist() if isinstance(population, np.ndarray) else population,
            "fitness_values": fitness_values.tolist() if isinstance(fitness_values, np.ndarray) else fitness_values,
            "best_position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
            "best_fitness": float(best_fitness),
            "optimizer_state": optimizer_state
        }
        
        self.log_data["checkpoints"].append(checkpoint_data)
        
        # Salva o checkpoint em um arquivo separado
        checkpoint_file = os.path.join(
            self.checkpoints_dir,
            f"{self.current_experiment}_checkpoint_iter_{iteration}_{phase}.json"
        )
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint salvo: {checkpoint_file}")
    
    def log_final_results(self, best_architecture, best_params, best_fitness, final_metrics):
        """Registra os resultados finais do experimento"""
        self.log_data["final_results"] = {
            "timestamp": datetime.now().isoformat(),
            "best_architecture": best_architecture,
            "best_params": best_params,
            "best_fitness": float(best_fitness),
            "final_metrics": final_metrics,
            "total_iterations": len(self.log_data["iterations"]),
            "total_evaluations": self.log_data["optimization_summary"]["total_evaluations"]
        }
        
        # Salva o log final
        self._save_log()
        
        # Exporta dados finais em CSV
        self.export_final_results()
    
    def _save_log(self):
        """Salva o log atual em um arquivo JSON."""
        if not self.current_experiment:
            return
            
        # Salva na pasta logs
        log_file = os.path.join(
            self.logs_dir,
            f"{self.current_experiment}_log.json"
        )
        
        # Função para lidar com valores especiais no JSON
        def json_encoder(obj):
            if isinstance(obj, float):
                if obj == float('inf'):
                    return "Infinity"
                elif obj == float('-inf'):
                    return "-Infinity"
                elif obj != obj:  # NaN
                    return "NaN"
            return obj
        
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2, default=json_encoder)
    
    def _create_summary_csv(self):
        """Cria arquivo CSV para resumo das avaliações"""
        if not self.current_experiment:
            return
            
        # Salva na pasta results diretamente
        summary_file = os.path.join(
            self.csv_dir,
            f"{self.current_experiment}_avaliacoes.csv"
        )
        
        fieldnames = [
            "iteration", "phase", "best_fitness", "architecture_type", 
            "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
            "total_params", "avg_inference_time", "memory_used_mb", "gflops",
            "architecture_config"
        ]
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def _update_summary_csv(self, iteration, phase, best_fitness, metrics, architecture_config):
        """Atualiza o arquivo CSV de resumo com nova iteração"""
        if not self.current_experiment:
            return
            
        # Salva na pasta results diretamente
        summary_file = os.path.join(
            self.csv_dir,
            f"{self.current_experiment}_avaliacoes.csv"
        )
        
        # Determina o tipo de arquitetura
        architecture_type = "Unknown"
        if architecture_config and "model_type" in architecture_config:
            architecture_type = architecture_config["model_type"]
        
        row = {
            "iteration": iteration,
            "phase": phase,
            "best_fitness": best_fitness,
            "architecture_type": architecture_type,
            "top1_acc": metrics.get("top1_acc", 0.0) if metrics else 0.0,
            "top5_acc": metrics.get("top5_acc", 0.0) if metrics else 0.0,
            "precision_macro": metrics.get("precision_macro", 0.0) if metrics else 0.0,
            "recall_macro": metrics.get("recall_macro", 0.0) if metrics else 0.0,
            "f1_macro": metrics.get("f1_macro", 0.0) if metrics else 0.0,
            "total_params": metrics.get("total_params", 0) if metrics else 0,
            "avg_inference_time": metrics.get("avg_inference_time", 0.0) if metrics else 0.0,
            "memory_used_mb": metrics.get("memory_used_mb", 0.0) if metrics else 0.0,
            "gflops": metrics.get("gflops", 0.0) if metrics else 0.0,
            "architecture_config": json.dumps(architecture_config) if architecture_config else ""
        }
        
        with open(summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            writer.writerow(row)
    
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
    
    def resume_from_checkpoint(self, checkpoint_file: str, optimizer: PSO) -> Tuple[int, str]:
        """
        Resume a otimização a partir de um checkpoint.
        
        Args:
            checkpoint_file (str): Caminho do arquivo de checkpoint
            optimizer (PSO): Instância do otimizador PSO
            
        Returns:
            Tuple[int, str]: (iteração, fase) para continuar a otimização
        """
        checkpoint_data = self.load_checkpoint(checkpoint_file)
        
        # Carrega o estado do otimizador
        if "optimizer_state" in checkpoint_data:
            optimizer.load_state_from_dict(checkpoint_data["optimizer_state"])
        
        return checkpoint_data["iteration"], checkpoint_data["phase"]
    
    def export_final_results(self):
        """Exporta resultados finais em múltiplos formatos"""
        if not self.current_experiment:
            return
            
        # Salva na pasta results diretamente
        results_dir = self.log_dir
        
        # 1. Exporta histórico de pbest/gbest
        self._export_pbest_gbest_history(results_dir)
        
        # 2. Exporta métricas detalhadas
        self._export_detailed_metrics(results_dir)
        
        # 3. Exporta resumo da otimização
        self._export_optimization_summary(results_dir)
        
        print(f"📊 Resultados exportados para: {results_dir}")
    
    def _export_pbest_gbest_history(self, results_dir: str):
        """Exporta histórico de pbest e gbest"""
        # Pbest history
        pbest_file = os.path.join(self.csv_dir, f"{self.current_experiment}_pbest_history.csv")
        if self.log_data["pbest_history"]:
            with open(pbest_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["iteration", "phase", "particle_id", "pbest_position", "pbest_cost"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in self.log_data["pbest_history"]:
                    for i, (pos, cost) in enumerate(zip(entry["pbest_positions"], entry["pbest_costs"])):
                        writer.writerow({
                            "iteration": entry["iteration"],
                            "phase": entry["phase"],
                            "particle_id": i,
                            "pbest_position": json.dumps(pos),
                            "pbest_cost": cost
                        })
        
        # Gbest history
        gbest_file = os.path.join(self.csv_dir, f"{self.current_experiment}_gbest_history.csv")
        if self.log_data["gbest_history"]:
            with open(gbest_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["iteration", "phase", "gbest_position", "gbest_cost"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in self.log_data["gbest_history"]:
                    writer.writerow({
                        "iteration": entry["iteration"],
                        "phase": entry["phase"],
                        "gbest_position": json.dumps(entry["gbest_position"]),
                        "gbest_cost": entry["gbest_cost"]
                    })
    
    def _export_detailed_metrics(self, results_dir: str):
        """Exporta métricas detalhadas"""
        metrics_file = os.path.join(self.csv_dir, f"{self.current_experiment}_detailed_metrics.csv")
        
        if self.log_data["iterations"]:
            with open(metrics_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["iteration", "phase", "best_fitness", "best_position", "metrics"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for iteration in self.log_data["iterations"]:
                    writer.writerow({
                        "iteration": iteration["iteration"],
                        "phase": iteration["phase"],
                        "best_fitness": iteration["best_fitness"],
                        "best_position": json.dumps(iteration["best_position"]),
                        "metrics": json.dumps(iteration.get("metrics", {}))
                    })
    
    def _export_optimization_summary(self, results_dir: str):
        """Exporta resumo da otimização"""
        summary_file = os.path.join(self.logs_dir, f"{self.current_experiment}_summary.json")
        
        summary_data = {
            "experiment_name": self.current_experiment,
            "optimization_summary": self.log_data["optimization_summary"],
            "best_solution": {
                "position": self.log_data["optimization_summary"]["best_position"],
                "fitness": self.log_data["optimization_summary"]["best_fitness"]
            },
            "total_checkpoints": len(self.log_data["checkpoints"]),
            "total_iterations": len(self.log_data["iterations"])
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        """Retorna a melhor solução encontrada"""
        if self.log_data["best_solutions"]:
            return self.log_data["best_solutions"][-1]
        return None
    
    def get_optimization_progress(self) -> Dict[str, Any]:
        """Retorna o progresso atual da otimização"""
        return {
            "total_iterations": len(self.log_data["iterations"]),
            "total_evaluations": self.log_data["optimization_summary"]["total_evaluations"],
            "best_score": self.log_data["optimization_summary"]["best_score"],
            "current_phase": self.log_data["iterations"][-1]["phase"] if self.log_data["iterations"] else None,
            "last_iteration": self.log_data["iterations"][-1]["iteration"] if self.log_data["iterations"] else 0
        }

# Exemplo de uso:
if __name__ == "__main__":
    # Criar instância do logger
    logger = OptimizationLogger()
    
    # Iniciar experimento
    config = {
        "population_size": 10,
        "max_iter": 50,
        "lambda_param": 0.5
    }
    logger.start_experiment(config)
    
    # Simular algumas iterações
    for i in range(5):
        logger.log_iteration(
            iteration=i+1,
            phase="PSO",
            population=np.random.rand(10, 5),
            fitness_values=np.random.rand(10),
            best_position=np.random.rand(5),
            best_fitness=0.8,
            metrics={"top1_acc": 0.85, "total_params": 1000000}
        )
    
    # Finalizar experimento
    logger.log_final_results("CNN", {"layers": 3}, 0.9, {"top1_acc": 0.9})
    
    print("✅ Logging concluído!") 