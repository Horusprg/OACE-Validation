import os
import sys
import json
import pandas as pd
import numpy as np
from utils.best_model_selection import load_results_data, select_best_architecture

def create_test_data():
    """Cria dados de teste para simular resultados de diferentes arquiteturas"""
    test_data = []
    
    # Cria 5 arquiteturas diferentes com métricas variadas
    for i in range(10):
        architecture = {
            'model': f'test_model_{i}',
            'architecture_config': {
                'layers': i + 1,
                'units': 64 * (i + 1)
            },
            'accuracy': np.random.uniform(0.6, 0.95),
            'precision': np.random.uniform(0.6, 0.95),
            'recall': np.random.uniform(0.6, 0.95),
            'total_params': np.random.uniform(1e5, 1e7),
            'avg_inference_time': np.random.uniform(0.001, 0.1),
            'memory_used_mb': np.random.uniform(40, 500),
            'gflops': np.random.uniform(0, 10)
        }
        test_data.append(architecture)
    
    return test_data

def test_best_model_selection():
    """Testa a seleção da melhor arquitetura"""
    # Cria diretório de teste
    test_dir = 'test_results'
    os.makedirs(test_dir, exist_ok=True)
    
    # Cria dados de teste
    test_data = create_test_data()
    
    print("test_data: ", test_data)
    
    # Salva dados de teste em arquivos JSON
    for i, data in enumerate(test_data):
        with open(os.path.join(test_dir, f'test_result_{i}.json'), 'w') as f:
            json.dump(data, f, indent=4)
    
    try:
        # Testa carregamento dos dados
        results_df = load_results_data(test_dir)
        print("\nDados carregados com sucesso!")
        print(f"Número de arquiteturas: {len(results_df)}")
        
        # Testa seleção da melhor arquitetura
        best_config, best_metrics = select_best_architecture(results_df)
        print("\nMelhor arquitetura selecionada com sucesso!")
        
        # Verifica se os arquivos de saída foram criados
        #assert os.path.exists('results_final.csv'), "Arquivo results_final.csv não foi criado"
        #assert os.path.exists('best_result.csv'), "Arquivo best_result.csv não foi criado"
        
        # Verifica se os dados foram salvos corretamente
        #results_final = pd.read_csv('results_final.csv')
        #best_result = pd.read_csv('best_result.csv')
        
        print("\nArquivos de saída criados com sucesso!")
        print(f"Resultados finais salvos em: results_final.csv")
        print(f"Melhor resultado salvo em: best_result.csv")
        
    except Exception as e:
        print(f"\nErro durante o teste: {str(e)}")
        raise
    
    finally:
        # Limpa arquivos de teste
        for file in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, file))
        os.rmdir(test_dir)
        
        # Limpa arquivos de saída
        if os.path.exists('results_final.csv'):
            os.remove('results_final.csv')
        if os.path.exists('best_result.csv'):
            os.remove('best_result.csv')

if __name__ == '__main__':
    test_best_model_selection() 