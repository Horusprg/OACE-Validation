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
    Classe aprimorada para registrar o progresso da otimização AFSA-GA-PSO.
    Armazena informações completas sobre cada iteração, incluindo:
    - Configurações detalhadas das arquiteturas
    - Métricas completas de assertividade e custo
    - Scores OACE calculados
    - Histórico completo de pbest e gbest
    - Checkpoints para retomada de experimentos
    - Dados estruturados para análise posterior
    """
    
    def __init__(self, log_dir: str = "results"):
        """
        Inicializa o logger de otimização aprimorado.
        
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
            "architecture_evaluations": [],  # Histórico completo de todas as arquiteturas avaliadas
            "pbest_history": [],             # Histórico detalhado de pbest
            "gbest_history": [],             # Histórico detalhado de gbest
            "oace_scores_history": [],       # Histórico de scores OACE calculados
            "optimization_summary": {
                "total_iterations": 0,
                "total_evaluations": 0,
                "total_architectures_tested": 0,
                "start_time": None,
                "end_time": None,
                "best_fitness": float('-inf'),  # Maximizar OACE
                "best_position": None,
                "best_oace_score": float('-inf'),
                "convergence_iteration": None,
                "optimization_phases": []
            }
        }
        
        # Estatísticas de cache e performance
        self.cache_stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "unique_architectures": set(),
            "evaluation_times": [],
            "phase_statistics": {}
        }
        
        # Criar estrutura de pastas
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Cria a estrutura de pastas organizadas"""
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiment_dir = None
        self.logs_dir = None
        self.csv_dir = None
        self.checkpoints_dir = None
        self.analysis_dir = None
    
    def start_experiment(self, config: Dict[str, Any]):
        """
        Inicia um novo experimento de otimização com configuração aprimorada.
        
        Args:
            config (Dict[str, Any]): Configuração completa do experimento
        """
        # Gera nome único para o experimento  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment = f"experiment_{timestamp}"
        
        # Cria estrutura de pastas organizada
        self.experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.csv_dir = os.path.join(self.experiment_dir, "csv")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.analysis_dir = os.path.join(self.experiment_dir, "analysis")
        
        for directory in [self.experiment_dir, self.logs_dir, self.csv_dir, 
                         self.checkpoints_dir, self.analysis_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Inicializa dados do experimento com informações detalhadas
        self.log_data["experiment_info"] = {
            "experiment_name": self.current_experiment,
            "start_time": datetime.now().isoformat(),
            "config": config,
            "system_info": {
                "python_version": f"{os.sys.version}",
                "working_directory": os.getcwd(),
                "log_directory": self.experiment_dir
            }
        }
        
        self.log_data["optimization_summary"]["start_time"] = datetime.now().isoformat()
        self.log_data["optimization_summary"]["best_fitness"] = float('-inf')
        
        # Cria arquivos CSV iniciais
        self._create_all_csv_files()
        
        print(f"🚀 Experimento iniciado: {self.current_experiment}")
        print(f"📁 Pasta do experimento: {self.experiment_dir}")
        print(f"   ├── 📄 logs/ (arquivos JSON detalhados)")
        print(f"   ├── 📊 csv/ (dados estruturados)")
        print(f"   ├── 💾 checkpoints/ (estados de otimização)")
        print(f"   └── 📈 analysis/ (análises e gráficos)")
    
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
                     architecture_config: Dict[str, Any] = None,
                     oace_score: float = None,
                     evaluation_time: float = None):
        """
        Registra uma iteração completa com todas as informações relevantes.
        """
        start_time = datetime.now()
        
        # Dados básicos da iteração
        iteration_data = {
            "iteration": iteration,
            "phase": phase,
            "timestamp": start_time.isoformat(),
            "population": population.tolist() if isinstance(population, np.ndarray) else population,
            "fitness_values": fitness_values.tolist() if isinstance(fitness_values, np.ndarray) else fitness_values,
            "best_position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
            "best_fitness": float(best_fitness),
            "metrics": metrics or {},
            "architecture_config": architecture_config or {},
            "oace_score": float(oace_score) if oace_score is not None else None,
            "evaluation_time": float(evaluation_time) if evaluation_time is not None else None
        }
        
        # Adiciona dados de pbest/gbest se fornecidos
        if pbest_pos is not None:
            iteration_data["pbest_positions"] = pbest_pos.tolist() if isinstance(pbest_pos, np.ndarray) else pbest_pos
        if pbest_cost is not None:
            iteration_data["pbest_costs"] = pbest_cost.tolist() if isinstance(pbest_cost, np.ndarray) else pbest_cost
        if gbest_pos is not None:
            iteration_data["gbest_position"] = gbest_pos.tolist() if isinstance(gbest_pos, np.ndarray) else gbest_pos
        if gbest_cost is not None:
            iteration_data["gbest_cost"] = float(gbest_cost)
            
        self.log_data["iterations"].append(iteration_data)
        
        # Registra como melhor solução se for melhor que a anterior
        if best_fitness > self.log_data["optimization_summary"]["best_fitness"]:
            best_solution = {
                "iteration": iteration,
                "phase": phase,
                "position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                "fitness": float(best_fitness),
                "metrics": metrics or {},
                "architecture_config": architecture_config or {},
                "oace_score": float(oace_score) if oace_score is not None else None,
                "timestamp": start_time.isoformat()
            }
            
            self.log_data["best_solutions"].append(best_solution)
            
            # CORREÇÃO: Atualiza o resumo da otimização com o score OACE correto
            self.log_data["optimization_summary"]["best_fitness"] = float(best_fitness)
            self.log_data["optimization_summary"]["best_position"] = best_position.tolist() if isinstance(best_position, np.ndarray) else best_position
            self.log_data["optimization_summary"]["convergence_iteration"] = iteration
            if oace_score is not None:
                self.log_data["optimization_summary"]["best_oace_score"] = float(oace_score)
            # CORREÇÃO: Se não há oace_score explícito, usa o best_fitness como fallback
            elif self.log_data["optimization_summary"]["best_oace_score"] == float('-inf'):
                self.log_data["optimization_summary"]["best_oace_score"] = float(best_fitness)
        
        # Registra histórico de métricas
        if metrics:
            self.log_data["metrics_history"].append({
                "iteration": iteration,
                "phase": phase,
                "metrics": metrics,
                "architecture_config": architecture_config or {},
                "timestamp": start_time.isoformat()
            })
        
        # Registra avaliação completa da arquitetura
        if architecture_config and metrics:
            arch_eval = {
                "iteration": iteration,
                "phase": phase,
                "architecture_config": architecture_config,
                "metrics": metrics,
                "fitness": float(best_fitness),
                "oace_score": float(oace_score) if oace_score is not None else None,
                "position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                "timestamp": start_time.isoformat(),
                "evaluation_id": f"{phase}_{iteration}_{len(self.log_data['architecture_evaluations'])}"
            }
            self.log_data["architecture_evaluations"].append(arch_eval)
            
            # Atualiza estatísticas
            arch_key = str(architecture_config)
            self.cache_stats["unique_architectures"].add(arch_key)
            self.log_data["optimization_summary"]["total_architectures_tested"] = len(self.cache_stats["unique_architectures"])
        
        # Registra histórico de OACE se disponível
        if oace_score is not None:
            self.log_data["oace_scores_history"].append({
                "iteration": iteration,
                "phase": phase,
                "oace_score": float(oace_score),
                "position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                "architecture_config": architecture_config or {},
                "timestamp": start_time.isoformat()
            })
        
        # Registra histórico detalhado de pbest/gbest
        if pbest_pos is not None and pbest_cost is not None:
            self.log_data["pbest_history"].append({
                "iteration": iteration,
                "phase": phase,
                "pbest_positions": pbest_pos.tolist() if isinstance(pbest_pos, np.ndarray) else pbest_pos,
                "pbest_costs": pbest_cost.tolist() if isinstance(pbest_cost, np.ndarray) else pbest_cost,
                "timestamp": start_time.isoformat()
            })
        
        if gbest_pos is not None and gbest_cost is not None:
            self.log_data["gbest_history"].append({
                "iteration": iteration,
                "phase": phase,
                "gbest_position": gbest_pos.tolist() if isinstance(gbest_pos, np.ndarray) else gbest_pos,
                "gbest_cost": float(gbest_cost),
                "timestamp": start_time.isoformat()
            })
        
        # Atualiza contadores
        self.log_data["optimization_summary"]["total_iterations"] = len(self.log_data["iterations"])
        self.log_data["optimization_summary"]["total_evaluations"] += 1
        
        # Salva o log e atualiza CSVs
        self._save_log()
        self._update_all_csv_files(iteration_data)
    
    def log_checkpoint(self, 
                      iteration: int,
                      phase: str,
                      population: np.ndarray,
                      fitness_values: np.ndarray,
                      best_position: np.ndarray,
                      best_fitness: float,
                      optimizer_state: Dict[str, Any] = None,
                      metadata: Dict[str, Any] = None):
        """
        Cria um checkpoint completo do estado atual da otimização.
        """
        checkpoint_data = {
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "population": population.tolist() if isinstance(population, np.ndarray) else population,
            "fitness_values": fitness_values.tolist() if isinstance(fitness_values, np.ndarray) else fitness_values,
            "best_position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
            "best_fitness": float(best_fitness),
            "optimizer_state": optimizer_state or {},
            "metadata": metadata or {},
            "cache_stats": self.cache_stats.copy(),
            "experiment_summary": self.log_data["optimization_summary"].copy()
        }
        
        self.log_data["checkpoints"].append({
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_file": f"checkpoint_iter_{iteration}_{phase}.json"
        })
        
        # Salva o checkpoint em arquivo separado
        checkpoint_file = os.path.join(
            self.checkpoints_dir,
            f"{self.current_experiment}_checkpoint_iter_{iteration}_{phase}.json"
        )
        
        def json_encoder(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, float):
                if obj == float('inf'): return "Infinity"
                elif obj == float('-inf'): return "-Infinity"
                elif obj != obj:  # NaN check
                    return "NaN"
            return obj
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=json_encoder)
        
        print(f"💾 Checkpoint salvo: {checkpoint_file}")
        return checkpoint_file
    
    def log_final_results(self, best_architecture, best_params, best_fitness, final_metrics):
        """Registra os resultados finais do experimento com informações completas"""
        end_time = datetime.now()
        
        # CORREÇÃO: Calcula o score OACE final corretamente
        best_oace_score = float(best_fitness) if best_fitness is not None else float('-inf')
        
        self.log_data["final_results"] = {
            "timestamp": end_time.isoformat(),
            "best_architecture": best_architecture,
            "best_params": best_params,
            "best_fitness": float(best_fitness),
            "final_metrics": final_metrics,
            "total_iterations": len(self.log_data["iterations"]),
            "total_evaluations": self.log_data["optimization_summary"]["total_evaluations"],
            "total_architectures_tested": len(self.cache_stats["unique_architectures"]),
            "experiment_duration": (end_time - datetime.fromisoformat(self.log_data["experiment_info"]["start_time"])).total_seconds(),
            "cache_efficiency": self.cache_stats["cache_hits"] / max(1, self.cache_stats["total_evaluations"]) * 100
        }
        
        # CORREÇÃO: Atualiza o resumo final com o score OACE correto
        self.log_data["optimization_summary"]["end_time"] = end_time.isoformat()
        self.log_data["optimization_summary"]["best_oace_score"] = best_oace_score
        
        # Salva o log final
        self._save_log()
        
        # Exporta todos os dados finais
        self.export_final_results()
    
    def _save_log(self):
        """Salva o log atual em um arquivo JSON com encoding apropriado."""
        if not self.current_experiment:
            return
            
        log_file = os.path.join(
            self.logs_dir,
            f"{self.current_experiment}_log.json"
        )
        
        def json_encoder(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, float):
                if obj == float('inf'):
                    return "Infinity"
                elif obj == float('-inf'):
                    return "-Infinity"
                elif obj != obj:  # NaN
                    return "NaN"
            return obj
        
        # CORREÇÃO: Garante que valores infinitos sejam tratados corretamente
        def clean_infinite_values(data):
            """Remove valores infinitos e NaN dos dados antes de salvar"""
            if isinstance(data, dict):
                cleaned = {}
                for key, value in data.items():
                    if isinstance(value, float):
                        if value == float('inf'):
                            cleaned[key] = "Infinity"
                        elif value == float('-inf'):
                            cleaned[key] = "-Infinity"
                        elif value != value:  # NaN
                            cleaned[key] = "NaN"
                        else:
                            cleaned[key] = value
                    elif isinstance(value, (dict, list)):
                        cleaned[key] = clean_infinite_values(value)
                    else:
                        cleaned[key] = value
                return cleaned
            elif isinstance(data, list):
                return [clean_infinite_values(item) for item in data]
            else:
                return data
        
        # CORREÇÃO: Limpa os dados antes de salvar
        cleaned_data = clean_infinite_values(self.log_data)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, default=json_encoder, ensure_ascii=False)
    
    def _create_all_csv_files(self):
        """Cria todos os arquivos CSV necessários para o experimento"""
        if not self.current_experiment:
            return
            
        # 1. Arquivo CSV para resumo das avaliações
        summary_file = os.path.join(
            self.csv_dir,
            f"{self.current_experiment}_avaliacoes.csv"
        )
        
        fieldnames = [
            "iteration", "phase", "best_fitness", "architecture_type", 
            "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
            "total_params", "avg_inference_time", "memory_used_mb", "gflops",
            "oace_score", "architecture_config"
        ]
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        # 2. Arquivo CSV para histórico detalhado de métricas
        metrics_file = os.path.join(
            self.csv_dir,
            f"{self.current_experiment}_detailed_metrics.csv"
        )
        
        metrics_fieldnames = [
            "iteration", "phase", "timestamp", "evaluation_id",
            "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
            "total_params", "avg_inference_time", "memory_used_mb", "gflops",
            "oace_score", "fitness", "model_type", "architecture_params"
        ]
        
        with open(metrics_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics_fieldnames)
            writer.writeheader()
        
        # 3. Arquivo CSV para histórico de OACE scores
        oace_file = os.path.join(
            self.csv_dir,
            f"{self.current_experiment}_oace_scores.csv"
        )
        
        oace_fieldnames = [
            "iteration", "phase", "timestamp", "oace_score", "fitness",
            "model_type", "position", "architecture_config"
        ]
        
        with open(oace_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=oace_fieldnames)
            writer.writeheader()
    
    def _update_all_csv_files(self, iteration_data):
        """Atualiza todos os arquivos CSV com nova iteração"""
        if not self.current_experiment:
            return
            
        # 1. Atualiza CSV de resumo das avaliações
        summary_file = os.path.join(
            self.csv_dir,
            f"{self.current_experiment}_avaliacoes.csv"
        )
        
        # Determina o tipo de arquitetura
        architecture_type = "Unknown"
        if iteration_data["architecture_config"] and "model_type" in iteration_data["architecture_config"]:
            architecture_type = iteration_data["architecture_config"]["model_type"]
        elif iteration_data["phase"] == "PSO":
            architecture_type = "PSO_Optimized"  # CORREÇÃO: Identifica como otimizado pelo PSO
        elif iteration_data["phase"] == "AFSA-PSO":
            architecture_type = "AFSA_Generated"  # CORREÇÃO: Identifica como gerado pelo AFSA
        elif iteration_data["phase"] == "GA-PSO":
            architecture_type = "GA_Optimized"  # CORREÇÃO: Identifica como otimizado pelo GA
        elif iteration_data["phase"] == "GA":
            architecture_type = "GA_Final"  # CORREÇÃO: Identifica como resultado final do GA
        
        row = {
            "iteration": iteration_data["iteration"],
            "phase": iteration_data["phase"],
            "best_fitness": iteration_data["best_fitness"],
            "architecture_type": architecture_type,
            "top1_acc": iteration_data["metrics"].get("top1_acc", 0.0) if iteration_data["metrics"] else 0.0,
            "top5_acc": iteration_data["metrics"].get("top5_acc", 0.0) if iteration_data["metrics"] else 0.0,
            "precision_macro": iteration_data["metrics"].get("precision_macro", 0.0) if iteration_data["metrics"] else 0.0,
            "recall_macro": iteration_data["metrics"].get("recall_macro", 0.0) if iteration_data["metrics"] else 0.0,
            "f1_macro": iteration_data["metrics"].get("f1_macro", 0.0) if iteration_data["metrics"] else 0.0,
            "total_params": iteration_data["metrics"].get("total_params", 0) if iteration_data["metrics"] else 0,
            "avg_inference_time": iteration_data["metrics"].get("avg_inference_time", 0.0) if iteration_data["metrics"] else 0.0,
            "memory_used_mb": iteration_data["metrics"].get("memory_used_mb", 0.0) if iteration_data["metrics"] else 0.0,
            "gflops": iteration_data["metrics"].get("gflops", 0.0) if iteration_data["metrics"] else 0.0,
            "oace_score": iteration_data.get("oace_score", ""),  # CORREÇÃO: Campo vazio para valores nulos
            "architecture_config": json.dumps(iteration_data["architecture_config"]) if iteration_data["architecture_config"] else ""
        }
        
        with open(summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            writer.writerow(row)
        
        # 2. Atualiza CSV de métricas detalhadas se há dados de arquitetura
        if iteration_data["metrics"]:  # CORREÇÃO: Remove dependência de architecture_config
            metrics_file = os.path.join(
                self.csv_dir,
                f"{self.current_experiment}_detailed_metrics.csv"
            )
            
            # CORREÇÃO: Determina o tipo de arquitetura mesmo sem architecture_config
            model_type = "Unknown"
            if iteration_data["architecture_config"] and "model_type" in iteration_data["architecture_config"]:
                model_type = iteration_data["architecture_config"]["model_type"]
            elif iteration_data["phase"] == "PSO":
                model_type = "PSO_Optimized"  # Identifica como otimizado pelo PSO
            elif iteration_data["phase"] == "AFSA-PSO":
                model_type = "AFSA_Generated"  # Identifica como gerado pelo AFSA
            elif iteration_data["phase"] == "GA-PSO":
                model_type = "GA_Optimized"  # Identifica como otimizado pelo GA
            elif iteration_data["phase"] == "GA":
                model_type = "GA_Final"  # Identifica como resultado final do GA
            
            metrics_row = {
                "iteration": iteration_data["iteration"],
                "phase": iteration_data["phase"],
                "timestamp": iteration_data["timestamp"],
                "evaluation_id": f"{iteration_data['phase']}_{iteration_data['iteration']}",
                "top1_acc": iteration_data["metrics"].get("top1_acc", 0.0),
                "top5_acc": iteration_data["metrics"].get("top5_acc", 0.0),
                "precision_macro": iteration_data["metrics"].get("precision_macro", 0.0),
                "recall_macro": iteration_data["metrics"].get("recall_macro", 0.0),
                "f1_macro": iteration_data["metrics"].get("f1_macro", 0.0),
                "total_params": iteration_data["metrics"].get("total_params", 0),
                "avg_inference_time": iteration_data["metrics"].get("avg_inference_time", 0.0),
                "memory_used_mb": iteration_data["metrics"].get("memory_used_mb", 0.0),
                "gflops": iteration_data["metrics"].get("gflops", 0.0),
                "oace_score": iteration_data.get("oace_score", ""),  # CORREÇÃO: Campo vazio para valores nulos
                "fitness": iteration_data["best_fitness"],
                "model_type": model_type,  # CORREÇÃO: Usa o tipo determinado acima
                "architecture_params": json.dumps(iteration_data["architecture_config"]) if iteration_data["architecture_config"] else "{}"
            }
            
            with open(metrics_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=metrics_row.keys())
                writer.writerow(metrics_row)
        
        # 3. Atualiza CSV de OACE scores se disponível
        if iteration_data.get("oace_score") is not None:
            oace_file = os.path.join(
                self.csv_dir,
                f"{self.current_experiment}_oace_scores.csv"
            )
            
            # CORREÇÃO: Determina o tipo de modelo corretamente
            model_type = "Unknown"
            if iteration_data["architecture_config"] and "model_type" in iteration_data["architecture_config"]:
                model_type = iteration_data["architecture_config"]["model_type"]
            elif iteration_data["phase"] == "PSO":
                model_type = "PSO_Optimized"
            elif iteration_data["phase"] == "AFSA-PSO":
                model_type = "AFSA_Generated"
            elif iteration_data["phase"] == "GA-PSO":
                model_type = "GA_Optimized"
            elif iteration_data["phase"] == "GA":
                model_type = "GA_Final"
            
            oace_row = {
                "iteration": iteration_data["iteration"],
                "phase": iteration_data["phase"],
                "timestamp": iteration_data["timestamp"],
                "oace_score": iteration_data["oace_score"],
                "fitness": iteration_data["best_fitness"],
                "model_type": model_type,  # CORREÇÃO: Usa o tipo determinado acima
                "position": json.dumps(iteration_data["best_position"]),
                "architecture_config": json.dumps(iteration_data["architecture_config"]) if iteration_data["architecture_config"] else "{}"
            }
            
            with open(oace_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=oace_row.keys())
                writer.writerow(oace_row)
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Carrega um checkpoint salvo com validação.
        
        Args:
            checkpoint_file (str): Caminho do arquivo de checkpoint
            
        Returns:
            Dict[str, Any]: Dados do checkpoint
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_file}")
            
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # Valida estrutura básica do checkpoint
        required_fields = ["iteration", "phase", "timestamp", "best_position", "best_fitness"]
        for field in required_fields:
            if field not in checkpoint_data:
                raise ValueError(f"Campo obrigatório '{field}' não encontrado no checkpoint")
        
        return checkpoint_data
    
    def resume_from_checkpoint(self, checkpoint_file: str, optimizer) -> Tuple[int, str]:
        """
        Resume a otimização a partir de um checkpoint com validação aprimorada.
        
        Args:
            checkpoint_file (str): Caminho do arquivo de checkpoint
            optimizer: Instância do otimizador (PSO, AFSA, etc.)
            
        Returns:
            Tuple[int, str]: (iteração, fase) para continuar a otimização
        """
        checkpoint_data = self.load_checkpoint(checkpoint_file)
        
        # Carrega o estado do otimizador se disponível
        if "optimizer_state" in checkpoint_data and hasattr(optimizer, 'load_state_from_dict'):
            optimizer.load_state_from_dict(checkpoint_data["optimizer_state"])
            print(f"🔄 Estado do otimizador carregado do checkpoint")
        
        # Restaura estatísticas de cache se disponíveis
        if "cache_stats" in checkpoint_data:
            self.cache_stats.update(checkpoint_data["cache_stats"])
            # Converte unique_architectures de volta para set se necessário
            if isinstance(self.cache_stats["unique_architectures"], list):
                self.cache_stats["unique_architectures"] = set(self.cache_stats["unique_architectures"])
        
        # Restaura resumo do experimento se disponível
        if "experiment_summary" in checkpoint_data:
            self.log_data["optimization_summary"].update(checkpoint_data["experiment_summary"])
        
        print(f"✅ Checkpoint carregado: Iteração {checkpoint_data['iteration']}, Fase {checkpoint_data['phase']}")
        return checkpoint_data["iteration"], checkpoint_data["phase"]
    
    def export_final_results(self):
        """Exporta resultados finais em múltiplos formatos organizados"""
        if not self.current_experiment:
            return
            
        print(f"📊 Exportando resultados finais...")
        
        # Exporta históricos específicos
        self._export_pbest_gbest_history()
        self._export_convergence_analysis()
        self._export_architecture_comparison()
        self._export_optimization_summary()
        
        print(f"✅ Todos os resultados exportados para: {self.experiment_dir}")
    
    def _export_pbest_gbest_history(self):
        """Exporta histórico completo de pbest e gbest"""
        # Pbest history
        if self.log_data["pbest_history"]:
            pbest_file = os.path.join(self.csv_dir, f"{self.current_experiment}_pbest_history.csv")
            with open(pbest_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["iteration", "phase", "timestamp", "particle_id", "pbest_position", "pbest_cost"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in self.log_data["pbest_history"]:
                    for i, (pos, cost) in enumerate(zip(entry["pbest_positions"], entry["pbest_costs"])):
                        writer.writerow({
                            "iteration": entry["iteration"],
                            "phase": entry["phase"],
                            "timestamp": entry["timestamp"],
                            "particle_id": i,
                            "pbest_position": json.dumps(pos),
                            "pbest_cost": cost
                        })
        
        # Gbest history
        if self.log_data["gbest_history"]:
            gbest_file = os.path.join(self.csv_dir, f"{self.current_experiment}_gbest_history.csv")
            with open(gbest_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["iteration", "phase", "timestamp", "gbest_position", "gbest_cost"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in self.log_data["gbest_history"]:
                    writer.writerow({
                        "iteration": entry["iteration"],
                        "phase": entry["phase"],
                        "timestamp": entry["timestamp"],
                        "gbest_position": json.dumps(entry["gbest_position"]),
                        "gbest_cost": entry["gbest_cost"]
                    })
    
    def _export_convergence_analysis(self):
        """Exporta análise de convergência"""
        if not self.log_data["iterations"]:
            return
            
        convergence_file = os.path.join(self.analysis_dir, f"{self.current_experiment}_convergence_analysis.csv")
        
        with open(convergence_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["iteration", "phase", "best_fitness", "oace_score", "improvement", "cumulative_improvement"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            previous_fitness = float('-inf')
            cumulative_improvement = 0
            
            for iteration in self.log_data["iterations"]:
                current_fitness = iteration["best_fitness"]
                improvement = current_fitness - previous_fitness if previous_fitness != float('-inf') else 0
                cumulative_improvement += max(0, improvement)
                
                # CORREÇÃO: Trata corretamente valores nulos de oace_score
                oace_score = iteration.get("oace_score")
                if oace_score is None:
                    oace_score = ""  # Campo vazio para valores nulos
                elif isinstance(oace_score, (int, float)):
                    oace_score = float(oace_score)  # Converte para float
                
                writer.writerow({
                    "iteration": iteration["iteration"],
                    "phase": iteration["phase"],
                    "best_fitness": current_fitness,
                    "oace_score": oace_score,  # CORREÇÃO: Usa o valor tratado
                    "improvement": improvement,
                    "cumulative_improvement": cumulative_improvement
                })
                
                previous_fitness = current_fitness
    
    def _export_architecture_comparison(self):
        """Exporta comparação detalhada entre arquiteturas testadas"""
        if not self.log_data["architecture_evaluations"]:
            return
            
        comparison_file = os.path.join(self.analysis_dir, f"{self.current_experiment}_architecture_comparison.csv")
        
        with open(comparison_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "evaluation_id", "iteration", "phase", "model_type", "fitness", "oace_score",
                "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
                "total_params", "avg_inference_time", "memory_used_mb", "gflops",
                "architecture_params", "timestamp"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for eval_data in self.log_data["architecture_evaluations"]:
                metrics = eval_data.get("metrics", {})
                arch_config = eval_data.get("architecture_config", {})
                
                writer.writerow({
                    "evaluation_id": eval_data.get("evaluation_id", ""),
                    "iteration": eval_data["iteration"],
                    "phase": eval_data["phase"],
                    "model_type": arch_config.get("model_type", "Unknown"),
                    "fitness": eval_data["fitness"],
                    "oace_score": eval_data.get("oace_score", 0),
                    "top1_acc": metrics.get("top1_acc", 0),
                    "top5_acc": metrics.get("top5_acc", 0),
                    "precision_macro": metrics.get("precision_macro", 0),
                    "recall_macro": metrics.get("recall_macro", 0),
                    "f1_macro": metrics.get("f1_macro", 0),
                    "total_params": metrics.get("total_params", 0),
                    "avg_inference_time": metrics.get("avg_inference_time", 0),
                    "memory_used_mb": metrics.get("memory_used_mb", 0),
                    "gflops": metrics.get("gflops", 0),
                    "architecture_params": json.dumps(arch_config),
                    "timestamp": eval_data.get("timestamp", "")
                })
    
    def _export_optimization_summary(self):
        """Exporta resumo completo da otimização"""
        summary_file = os.path.join(self.analysis_dir, f"{self.current_experiment}_optimization_summary.json")
        
        # Calcula estatísticas adicionais
        iterations_by_phase = {}
        fitness_by_phase = {}
        
        for iteration in self.log_data["iterations"]:
            phase = iteration["phase"]
            if phase not in iterations_by_phase:
                iterations_by_phase[phase] = 0
                fitness_by_phase[phase] = []
            
            iterations_by_phase[phase] += 1
            fitness_by_phase[phase].append(iteration["best_fitness"])
        
        # Calcula estatísticas por fase
        phase_stats = {}
        for phase, fitness_values in fitness_by_phase.items():
            if fitness_values:
                phase_stats[phase] = {
                    "iterations": iterations_by_phase[phase],
                    "best_fitness": max(fitness_values),
                    "worst_fitness": min(fitness_values),
                    "avg_fitness": sum(fitness_values) / len(fitness_values),
                    "fitness_improvement": max(fitness_values) - min(fitness_values) if len(fitness_values) > 1 else 0
                }
        
        summary_data = {
            "experiment_info": self.log_data["experiment_info"],
            "optimization_summary": self.log_data["optimization_summary"],
            "phase_statistics": phase_stats,
            "cache_statistics": {
                "total_evaluations": self.cache_stats["total_evaluations"],
                "cache_hits": self.cache_stats["cache_hits"],
                "cache_misses": self.cache_stats["cache_misses"],
                "cache_efficiency": self.cache_stats["cache_hits"] / max(1, self.cache_stats["total_evaluations"]) * 100,
                "unique_architectures_tested": len(self.cache_stats["unique_architectures"])
            },
            "final_results": self.log_data.get("final_results", {}),
            "best_solution": self.get_best_solution()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        """Retorna informações sobre a melhor solução encontrada"""
        if not self.log_data["best_solutions"]:
            return None
        
        # Retorna a melhor solução (última na lista, que é a melhor)
        best = self.log_data["best_solutions"][-1]
        return {
            "iteration": best["iteration"],
            "phase": best["phase"],
            "fitness": best["fitness"],
            "oace_score": best.get("oace_score"),
            "metrics": best["metrics"],
            "architecture_config": best["architecture_config"],
            "position": best["position"],
            "timestamp": best.get("timestamp")
        }
    
    def get_optimization_progress(self) -> Dict[str, Any]:
        """Retorna informações sobre o progresso da otimização"""
        if not self.log_data["iterations"]:
            return {"status": "no_iterations"}
        
        last_iteration = self.log_data["iterations"][-1]
        
        return {
            "current_iteration": last_iteration["iteration"],
            "current_phase": last_iteration["phase"],
            "total_iterations": len(self.log_data["iterations"]),
            "total_evaluations": self.log_data["optimization_summary"]["total_evaluations"],
            "total_architectures_tested": self.log_data["optimization_summary"]["total_architectures_tested"],
            "best_fitness": self.log_data["optimization_summary"]["best_fitness"],
            "best_oace_score": self.log_data["optimization_summary"].get("best_oace_score"),
            "convergence_iteration": self.log_data["optimization_summary"].get("convergence_iteration"),
            "cache_efficiency": self.cache_stats["cache_hits"] / max(1, self.cache_stats["total_evaluations"]) * 100,
            "last_update": last_iteration["timestamp"]
        }

    def add_cache_hit(self, architecture_key: str):
        """Registra um cache hit"""
        self.cache_stats["cache_hits"] += 1
        self.cache_stats["total_evaluations"] += 1

    def add_cache_miss(self, architecture_key: str, evaluation_time: float = None):
        """Registra um cache miss"""
        self.cache_stats["cache_misses"] += 1
        self.cache_stats["total_evaluations"] += 1
        if evaluation_time is not None:
            self.cache_stats["evaluation_times"].append(evaluation_time)

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