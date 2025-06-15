import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.afsa_ga_pso import AFSAGAPSO
from utils.data_loader import get_cifar10_dataloaders
from utils.log_analyzer import OptimizationLogAnalyzer
import matplotlib.pyplot as plt

def main():
    # Carrega os dados
    train_loader, val_loader, test_loader = get_cifar10_dataloaders()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Configura o otimizador
    optimizer = AFSAGAPSO(
        population_size=10,
        max_iter=50,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        lambda_param=0.5,
        log_dir="results"
    )
    
    # Executa a otimização
    print("Iniciando otimização...")
    best_architecture, best_params, best_fitness = optimizer.optimize()
    
    print("\nOtimização concluída!")
    print(f"Melhor arquitetura: {best_architecture}")
    print(f"Score OACE final: {best_fitness:.6f}")
    
    # Analisa os logs
    print("\nAnalisando logs...")
    analyzer = OptimizationLogAnalyzer(log_dir="results")
    
    # Lista os experimentos disponíveis
    experiments = analyzer.list_experiments()
    print(f"\nExperimentos disponíveis: {experiments}")
    
    if experiments:
        # Plota o histórico de fitness
        print("\nGerando gráfico de fitness...")
        analyzer.plot_fitness_history(
            experiments[-1],
            save_path=os.path.join("results", experiments[-1], "fitness_history.png")
        )
        
        # Plota o histórico de métricas
        print("\nGerando gráfico de métricas...")
        analyzer.plot_metrics_history(
            experiments[-1],
            save_path=os.path.join("results", experiments[-1], "metrics_history.png")
        )
        
        # Gera resumo das melhores soluções
        print("\nGerando resumo das melhores soluções...")
        summary = analyzer.generate_summary(experiments[-1])
        print("\nResumo das melhores soluções:")
        print(summary)
        
        # Salva o resumo em CSV
        summary_path = os.path.join("results", experiments[-1], "best_solutions_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nResumo salvo em: {summary_path}")

if __name__ == "__main__":
    main()