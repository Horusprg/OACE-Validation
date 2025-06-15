import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import pandas as pd

class OptimizationLogAnalyzer:
    """
    Classe para analisar e visualizar os logs de otimização do algoritmo AFSA-GA-PSO.
    """
    
    def __init__(self, log_dir: str = "results"):
        """
        Inicializa o analisador de logs.
        
        Args:
            log_dir (str): Diretório onde os logs estão salvos
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def list_experiments(self) -> List[str]:
        """
        Lista todos os experimentos disponíveis.
        
        Returns:
            List[str]: Lista de IDs dos experimentos
        """
        if not os.path.exists(self.log_dir):
            return []
            
        experiments = []
        for d in os.listdir(self.log_dir):
            exp_dir = os.path.join(self.log_dir, d)
            if os.path.isdir(exp_dir):
                log_file = os.path.join(exp_dir, "optimization_log.json")
                if os.path.exists(log_file):
                    experiments.append(d)
        return experiments
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Carrega os dados de um experimento.
        
        Args:
            experiment_id (str): ID do experimento
            
        Returns:
            Dict[str, Any]: Dados do experimento
            
        Raises:
            FileNotFoundError: Se o arquivo de log não existir
            ValueError: Se o experimento não existir
        """
        if not os.path.exists(self.log_dir):
            raise ValueError(f"Diretório de logs '{self.log_dir}' não existe")
            
        exp_dir = os.path.join(self.log_dir, experiment_id)
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experimento '{experiment_id}' não encontrado")
            
        log_file = os.path.join(exp_dir, "optimization_log.json")
        if not os.path.exists(log_file):
            raise FileNotFoundError(
                f"Arquivo de log não encontrado para o experimento '{experiment_id}'. "
                f"Verifique se o experimento foi executado corretamente."
            )
            
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Erro ao decodificar o arquivo de log do experimento '{experiment_id}'")
    
    def plot_fitness_history(self, experiment_id: str, save_path: str = None):
        """
        Plota o histórico de fitness ao longo das iterações.
        
        Args:
            experiment_id (str): ID do experimento
            save_path (str, optional): Caminho para salvar o gráfico
            
        Raises:
            ValueError: Se não houver dados suficientes para plotar
        """
        try:
            data = self.load_experiment(experiment_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"Erro ao carregar experimento: {str(e)}")
            return
            
        if not data.get("iterations"):
            print(f"Nenhum dado de iteração encontrado para o experimento '{experiment_id}'")
            return
        
        # Organiza os dados por fase
        phases = {}
        for iteration in data["iterations"]:
            phase = iteration["phase"]
            if phase not in phases:
                phases[phase] = {
                    "iterations": [],
                    "best_fitness": []
                }
            phases[phase]["iterations"].append(iteration["iteration"])
            phases[phase]["best_fitness"].append(iteration["best_fitness"])
        
        if not phases:
            print(f"Nenhum dado de fase encontrado para o experimento '{experiment_id}'")
            return
        
        # Plota o gráfico
        plt.figure(figsize=(12, 6))
        for phase, values in phases.items():
            plt.plot(values["iterations"], values["best_fitness"], 
                    label=phase, marker='o', markersize=4)
        
        plt.title("Histórico de Fitness por Fase")
        plt.xlabel("Iteração")
        plt.ylabel("Fitness (Score OACE)")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"Gráfico salvo em: {save_path}")
            except Exception as e:
                print(f"Erro ao salvar gráfico: {str(e)}")
        plt.show()
    
    def plot_metrics_history(self, experiment_id: str, save_path: str = None):
        """
        Plota o histórico das métricas ao longo das iterações.
        
        Args:
            experiment_id (str): ID do experimento
            save_path (str, optional): Caminho para salvar o gráfico
            
        Raises:
            ValueError: Se não houver dados suficientes para plotar
        """
        try:
            data = self.load_experiment(experiment_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"Erro ao carregar experimento: {str(e)}")
            return
            
        if not data.get("iterations"):
            print(f"Nenhum dado de iteração encontrado para o experimento '{experiment_id}'")
            return
        
        # Organiza as métricas
        metrics = {
            "top1_acc": [],
            "top5_acc": [],
            "precision_macro": [],
            "recall_macro": [],
            "f1_macro": [],
            "total_params": [],
            "avg_inference_time": [],
            "memory_used_mb": [],
            "gflops": []
        }
        
        iterations = []
        for iteration in data["iterations"]:
            if not iteration.get("metrics"):
                continue
            iterations.append(iteration["iteration"])
            for metric in metrics:
                if metric in iteration["metrics"]:
                    metrics[metric].append(iteration["metrics"][metric])
                else:
                    metrics[metric].append(None)
        
        if not iterations:
            print(f"Nenhum dado de métrica encontrado para o experimento '{experiment_id}'")
            return
        
        # Plota os gráficos
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Métricas de assertividade
        for metric in ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]:
            if any(v is not None for v in metrics[metric]):
                ax1.plot(iterations, metrics[metric], label=metric, marker='o', markersize=4)
        
        ax1.set_title("Métricas de Assertividade")
        ax1.set_xlabel("Iteração")
        ax1.set_ylabel("Valor")
        ax1.legend()
        ax1.grid(True)
        
        # Métricas de custo
        for metric in ["total_params", "avg_inference_time", "memory_used_mb", "gflops"]:
            if any(v is not None for v in metrics[metric]):
                ax2.plot(iterations, metrics[metric], label=metric, marker='o', markersize=4)
        
        ax2.set_title("Métricas de Custo")
        ax2.set_xlabel("Iteração")
        ax2.set_ylabel("Valor")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"Gráfico salvo em: {save_path}")
            except Exception as e:
                print(f"Erro ao salvar gráfico: {str(e)}")
        plt.show()
    
    def generate_summary(self, experiment_id: str) -> pd.DataFrame:
        """
        Gera um resumo das melhores soluções encontradas.
        
        Args:
            experiment_id (str): ID do experimento
            
        Returns:
            pd.DataFrame: DataFrame com o resumo das soluções
            
        Raises:
            ValueError: Se não houver dados suficientes para gerar o resumo
        """
        try:
            data = self.load_experiment(experiment_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"Erro ao carregar experimento: {str(e)}")
            return pd.DataFrame()
            
        if not data.get("best_solutions"):
            print(f"Nenhuma solução encontrada para o experimento '{experiment_id}'")
            return pd.DataFrame()
        
        # Organiza os dados das melhores soluções
        summary_data = []
        for solution in data["best_solutions"]:
            if not solution.get("metrics"):
                continue
            summary_data.append({
                "iteration": solution["iteration"],
                "phase": solution["phase"],
                "fitness": solution["fitness"],
                **solution["metrics"]
            })
        
        if not summary_data:
            print(f"Nenhum dado válido encontrado para o experimento '{experiment_id}'")
            return pd.DataFrame()
            
        return pd.DataFrame(summary_data)
    
    def compare_experiments(self, experiment_ids: List[str], metric: str = "fitness"):
        """
        Compara múltiplos experimentos.
        
        Args:
            experiment_ids (List[str]): Lista de IDs dos experimentos
            metric (str): Métrica para comparação
        """
        plt.figure(figsize=(12, 6))
        
        for exp_id in experiment_ids:
            try:
                data = self.load_experiment(exp_id)
            except (FileNotFoundError, ValueError) as e:
                print(f"Erro ao carregar experimento '{exp_id}': {str(e)}")
                continue
                
            if not data.get("iterations"):
                print(f"Nenhum dado de iteração encontrado para o experimento '{exp_id}'")
                continue
            
            # Organiza os dados
            iterations = []
            values = []
            for iteration in data["iterations"]:
                iterations.append(iteration["iteration"])
                if metric == "fitness":
                    values.append(iteration["best_fitness"])
                elif iteration.get("metrics") and metric in iteration["metrics"]:
                    values.append(iteration["metrics"][metric])
                else:
                    values.append(None)
            
            if any(v is not None for v in values):
                plt.plot(iterations, values, label=exp_id, marker='o', markersize=4)
        
        plt.title(f"Comparação de Experimentos - {metric}")
        plt.xlabel("Iteração")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show() 