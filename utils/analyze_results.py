#!/usr/bin/env python3
"""
Script para análise e visualização dos resultados da otimização OACE.

Este script fornece ferramentas para:
- Carregar resultados salvos dos experimentos
- Visualizar convergência do score OACE
- Analisar distribuição dos parâmetros estruturais
- Comparar métricas de desempenho entre execuções
- Gerar relatórios de análise

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração do matplotlib para português
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

class OACEAnalyzer:
    """
    Classe para análise e visualização dos resultados da otimização OACE.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Inicializa o analisador.
        
        Args:
            results_dir (str): Diretório onde estão os resultados
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.loaded_data = {}
        
    def discover_experiments(self) -> List[str]:
        """
        Descobre todos os experimentos disponíveis.
        
        Returns:
            List[str]: Lista de IDs dos experimentos
        """
        experiments = []
        
        # Procura por experimentos em subdiretórios
        for exp_dir in self.results_dir.glob("**/experiment_*"):
            if exp_dir.is_dir():
                exp_id = exp_dir.name
                experiments.append(exp_id)
                
        return sorted(experiments)
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Carrega os dados de um experimento específico.
        
        Args:
            experiment_id (str): ID do experimento
            
        Returns:
            Dict[str, Any]: Dados do experimento
        """
        if experiment_id in self.loaded_data:
            return self.loaded_data[experiment_id]
        
        # Procura pelo arquivo de log do experimento
        log_file = None
        for exp_dir in self.results_dir.glob("**/experiment_*"):
            if exp_dir.name == experiment_id:
                log_file = exp_dir / "optimization_log.json"
                break
        
        if not log_file or not log_file.exists():
            raise FileNotFoundError(f"Log do experimento {experiment_id} não encontrado")
        
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        self.loaded_data[experiment_id] = data
        return data
    
    def load_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """
        Carrega todos os experimentos disponíveis.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dicionário com todos os experimentos
        """
        experiments = self.discover_experiments()
        
        for exp_id in experiments:
            try:
                self.load_experiment(exp_id)
            except Exception as e:
                print(f"Erro ao carregar experimento {exp_id}: {e}")
        
        return self.loaded_data
    
    def plot_oace_convergence(self, 
                             experiment_ids: Optional[List[str]] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plota a convergência do score OACE ao longo das iterações.
        
        Args:
            experiment_ids (List[str]): IDs dos experimentos para plotar
            save_path (str): Caminho para salvar o gráfico
            figsize (Tuple): Tamanho da figura
        """
        if experiment_ids is None:
            experiment_ids = list(self.loaded_data.keys())
        
        if not experiment_ids:
            print("Nenhum experimento carregado. Execute load_all_experiments() primeiro.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiment_ids)))
        
        for i, exp_id in enumerate(experiment_ids):
            if exp_id not in self.loaded_data:
                continue
                
            data = self.loaded_data[exp_id]
            iterations = data.get('iterations', [])
            
            if not iterations:
                continue
            
            # Extrai dados de convergência
            iters = []
            gbest_scores = []
            pbest_scores = []
            
            for iter_data in iterations:
                iters.append(iter_data['iteration'])
                gbest_scores.append(iter_data['best_fitness'])
                
                # Calcula pbest (melhor da população atual)
                fitness_values = iter_data.get('fitness_values', [])
                if fitness_values:
                    pbest_scores.append(max(fitness_values))
                else:
                    pbest_scores.append(iter_data['best_fitness'])
            
            # Plota gbest
            ax1.plot(iters, gbest_scores, 
                    color=colors[i], 
                    label=f'{exp_id} (GBest)',
                    linewidth=2,
                    alpha=0.8)
            
            # Plota pbest
            ax2.plot(iters, pbest_scores, 
                    color=colors[i], 
                    label=f'{exp_id} (PBest)',
                    linewidth=2,
                    alpha=0.8)
        
        # Configuração dos gráficos
        ax1.set_title('Convergência do Score OACE - GBest', fontweight='bold')
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Score OACE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Convergência do Score OACE - PBest', fontweight='bold')
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Score OACE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_parameter_distribution(self, 
                                   experiment_ids: Optional[List[str]] = None,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plota a distribuição dos parâmetros estruturais das arquiteturas otimizadas.
        
        Args:
            experiment_ids (List[str]): IDs dos experimentos para plotar
            save_path (str): Caminho para salvar o gráfico
            figsize (Tuple): Tamanho da figura
        """
        if experiment_ids is None:
            experiment_ids = list(self.loaded_data.keys())
        
        if not experiment_ids:
            print("Nenhum experimento carregado. Execute load_all_experiments() primeiro.")
            return
        
        # Coleta todos os parâmetros
        all_params = {}
        
        for exp_id in experiment_ids:
            if exp_id not in self.loaded_data:
                continue
                
            data = self.loaded_data[exp_id]
            iterations = data.get('iterations', [])
            
            for iter_data in iterations:
                position = iter_data.get('best_position', [])
                if position:
                    for i, param_value in enumerate(position):
                        param_name = f'Parâmetro_{i+1}'
                        if param_name not in all_params:
                            all_params[param_name] = []
                        all_params[param_name].append(param_value)
        
        if not all_params:
            print("Nenhum parâmetro encontrado nos dados.")
            return
        
        # Cria subplots para cada parâmetro
        n_params = len(all_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for i, (param_name, values) in enumerate(all_params.items()):
            ax = axes[i]
            
            # Histograma
            ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribuição de {param_name}', fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Adiciona estatísticas
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Média: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', 
                      label=f'+1σ: {mean_val + std_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', 
                      label=f'-1σ: {mean_val - std_val:.3f}')
            ax.legend()
        
        # Remove subplots vazios
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Distribuição dos Parâmetros Estruturais das Arquiteturas Otimizadas', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, 
                               experiment_ids: Optional[List[str]] = None,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Compara as métricas de desempenho entre diferentes execuções.
        
        Args:
            experiment_ids (List[str]): IDs dos experimentos para comparar
            save_path (str): Caminho para salvar o gráfico
            figsize (Tuple): Tamanho da figura
        """
        if experiment_ids is None:
            experiment_ids = list(self.loaded_data.keys())
        
        if not experiment_ids:
            print("Nenhum experimento carregado. Execute load_all_experiments() primeiro.")
            return
        
        # Coleta métricas finais de cada experimento
        metrics_data = {
            'experiment': [],
            'oace_score': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'cost_mtp': [],
            'cost_tpi': [],
            'cost_ms': []
        }
        
        for exp_id in experiment_ids:
            if exp_id not in self.loaded_data:
                continue
                
            data = self.loaded_data[exp_id]
            final_results = data.get('final_results', {})
            
            if not final_results:
                # Tenta pegar a melhor solução das iterações
                iterations = data.get('iterations', [])
                if iterations:
                    best_iter = max(iterations, key=lambda x: x['best_fitness'])
                    final_results = {
                        'best_fitness': best_iter['best_fitness'],
                        'final_metrics': best_iter.get('metrics', {})
                    }
            
            if final_results:
                metrics_data['experiment'].append(exp_id)
                metrics_data['oace_score'].append(final_results.get('best_fitness', 0))
                
                final_metrics = final_results.get('final_metrics', {})
                metrics_data['accuracy'].append(final_metrics.get('accuracy', 0))
                metrics_data['precision'].append(final_metrics.get('precision', 0))
                metrics_data['recall'].append(final_metrics.get('recall', 0))
                metrics_data['cost_mtp'].append(final_metrics.get('MTP', 0))
                metrics_data['cost_tpi'].append(final_metrics.get('TPI', 0))
                metrics_data['cost_ms'].append(final_metrics.get('MS', 0))
        
        if not metrics_data['experiment']:
            print("Nenhuma métrica encontrada nos dados.")
            return
        
        # Cria DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Cria subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Métricas de assertividade
        metrics_to_plot = [
            ('oace_score', 'Score OACE'),
            ('accuracy', 'Acurácia'),
            ('precision', 'Precisão'),
            ('recall', 'Recall'),
            ('cost_mtp', 'Custo MTP'),
            ('cost_ms', 'Custo MS')
        ]
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            bars = ax.bar(range(len(df)), df[metric], 
                         color=plt.cm.Set3(i/len(metrics_to_plot)), alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Experimento')
            ax.set_ylabel('Valor')
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([exp.split('_')[-1] for exp in df['experiment']], 
                              rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Adiciona valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Comparação de Métricas de Desempenho entre Execuções', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        plt.show()
        
        # Retorna o DataFrame para análise adicional
        return df
    
    def plot_phase_analysis(self, 
                           experiment_ids: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Analisa o desempenho por fase do algoritmo (AFSA, GA, PSO).
        
        Args:
            experiment_ids (List[str]): IDs dos experimentos para analisar
            save_path (str): Caminho para salvar o gráfico
            figsize (Tuple): Tamanho da figura
        """
        if experiment_ids is None:
            experiment_ids = list(self.loaded_data.keys())
        
        if not experiment_ids:
            print("Nenhum experimento carregado. Execute load_all_experiments() primeiro.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        phase_data = {'AFSA': [], 'GA': [], 'PSO': []}
        phase_iterations = {'AFSA': [], 'GA': [], 'PSO': []}
        
        for exp_id in experiment_ids:
            if exp_id not in self.loaded_data:
                continue
                
            data = self.loaded_data[exp_id]
            iterations = data.get('iterations', [])
            
            current_phase = None
            phase_scores = []
            
            for iter_data in iterations:
                phase = iter_data.get('phase', 'Unknown')
                score = iter_data['best_fitness']
                
                if phase in phase_data:
                    phase_data[phase].append(score)
                    phase_iterations[phase].append(iter_data['iteration'])
        
        # Gráfico de evolução por fase
        colors = {'AFSA': 'blue', 'GA': 'green', 'PSO': 'red'}
        
        for phase, scores in phase_data.items():
            if scores:
                ax1.plot(range(len(scores)), scores, 
                        color=colors[phase], 
                        label=phase, 
                        linewidth=2,
                        marker='o', markersize=4)
        
        ax1.set_title('Evolução do Score OACE por Fase', fontweight='bold')
        ax1.set_xlabel('Iteração da Fase')
        ax1.set_ylabel('Score OACE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Boxplot por fase
        phase_scores_list = [scores for scores in phase_data.values() if scores]
        phase_names = [phase for phase, scores in phase_data.items() if scores]
        
        if phase_scores_list:
            box_plot = ax2.boxplot(phase_scores_list, labels=phase_names, patch_artist=True)
            
            # Cores diferentes para cada fase
            for patch, color in zip(box_plot['boxes'], colors.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title('Distribuição de Scores por Fase', fontweight='bold')
            ax2.set_ylabel('Score OACE')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, 
                               experiment_ids: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> str:
        """
        Gera um relatório resumido dos experimentos.
        
        Args:
            experiment_ids (List[str]): IDs dos experimentos para incluir
            save_path (str): Caminho para salvar o relatório
            
        Returns:
            str: Conteúdo do relatório
        """
        if experiment_ids is None:
            experiment_ids = list(self.loaded_data.keys())
        
        if not experiment_ids:
            return "Nenhum experimento carregado."
        
        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE ANÁLISE DOS EXPERIMENTOS OACE")
        report.append("=" * 60)
        report.append(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append(f"Total de experimentos analisados: {len(experiment_ids)}")
        report.append("")
        
        # Estatísticas gerais
        all_scores = []
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.loaded_data:
                continue
                
            data = self.loaded_data[exp_id]
            iterations = data.get('iterations', [])
            
            if iterations:
                best_score = max(iter['best_fitness'] for iter in iterations)
                all_scores.append(best_score)
                
                # Tenta encontrar métricas finais
                final_results = data.get('final_results', {})
                if final_results:
                    metrics = final_results.get('final_metrics', {})
                    all_accuracies.append(metrics.get('accuracy', 0))
                    all_precisions.append(metrics.get('precision', 0))
                    all_recalls.append(metrics.get('recall', 0))
        
        if all_scores:
            report.append("ESTATÍSTICAS GERAIS:")
            report.append("-" * 30)
            report.append(f"Score OACE médio: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
            report.append(f"Score OACE máximo: {np.max(all_scores):.4f}")
            report.append(f"Score OACE mínimo: {np.min(all_scores):.4f}")
            report.append("")
        
        if all_accuracies:
            report.append("MÉTRICAS DE ASSERTIVIDADE:")
            report.append("-" * 30)
            report.append(f"Acurácia média: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
            report.append(f"Precisão média: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
            report.append(f"Recall médio: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
            report.append("")
        
        # Detalhes por experimento
        report.append("DETALHES POR EXPERIMENTO:")
        report.append("-" * 30)
        
        for exp_id in experiment_ids:
            if exp_id not in self.loaded_data:
                continue
                
            data = self.loaded_data[exp_id]
            iterations = data.get('iterations', [])
            
            if iterations:
                best_score = max(iter['best_fitness'] for iter in iterations)
                total_iterations = len(iterations)
                
                report.append(f"Experimento: {exp_id}")
                report.append(f"  - Total de iterações: {total_iterations}")
                report.append(f"  - Melhor score OACE: {best_score:.4f}")
                
                # Informações sobre fases
                phases = set(iter.get('phase', 'Unknown') for iter in iterations)
                report.append(f"  - Fases executadas: {', '.join(phases)}")
                report.append("")
        
        report_content = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Relatório salvo em: {save_path}")
        
        return report_content
    
    def create_dashboard(self, 
                        experiment_ids: Optional[List[str]] = None,
                        save_dir: str = "analysis_results") -> None:
        """
        Cria um dashboard completo com todos os gráficos e relatórios.
        
        Args:
            experiment_ids (List[str]): IDs dos experimentos para analisar
            save_dir (str): Diretório para salvar os resultados
        """
        if experiment_ids is None:
            experiment_ids = list(self.loaded_data.keys())
        
        if not experiment_ids:
            print("Nenhum experimento carregado. Execute load_all_experiments() primeiro.")
            return
        
        # Cria diretório de saída
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print("🔍 Criando dashboard de análise...")
        
        # Gera todos os gráficos
        self.plot_oace_convergence(
            experiment_ids, 
            save_path / "01_convergencia_oace.png"
        )
        
        self.plot_parameter_distribution(
            experiment_ids, 
            save_path / "02_distribuicao_parametros.png"
        )
        
        self.plot_metrics_comparison(
            experiment_ids, 
            save_path / "03_comparacao_metricas.png"
        )
        
        self.plot_phase_analysis(
            experiment_ids, 
            save_path / "04_analise_fases.png"
        )
        
        # Gera relatório
        self.generate_summary_report(
            experiment_ids, 
            save_path / "05_relatorio_resumo.txt"
        )
        
        print(f"✅ Dashboard criado com sucesso em: {save_path}")
        print("📊 Arquivos gerados:")
        print("   - 01_convergencia_oace.png")
        print("   - 02_distribuicao_parametros.png")
        print("   - 03_comparacao_metricas.png")
        print("   - 04_analise_fases.png")
        print("   - 05_relatorio_resumo.txt")


def main():
    """
    Função principal para execução do script.
    """
    print("🔬 ANALISADOR DE RESULTADOS OACE")
    print("=" * 50)
    
    # Inicializa o analisador
    analyzer = OACEAnalyzer()
    
    # Descobre experimentos
    experiments = analyzer.discover_experiments()
    
    if not experiments:
        print("❌ Nenhum experimento encontrado no diretório 'results'.")
        print("Execute alguns experimentos primeiro usando main.py")
        return
    
    print(f"📁 Experimentos encontrados: {len(experiments)}")
    for exp in experiments:
        print(f"   - {exp}")
    
    # Carrega todos os experimentos
    print("\n📂 Carregando dados dos experimentos...")
    analyzer.load_all_experiments()
    
    # Cria dashboard completo
    print("\n📊 Criando dashboard de análise...")
    analyzer.create_dashboard()
    
    print("\n✅ Análise concluída!")
    print("📈 Verifique os gráficos e relatórios gerados na pasta 'analysis_results'")


if __name__ == "__main__":
    main() 