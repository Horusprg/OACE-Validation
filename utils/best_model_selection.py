import json
import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from utils.oace_evaluation import calculate_oace_score
from utils.ahp_weights import calculate_ahp_weights, dataset_a, dataset_c, weight_derivation

def load_results_data(results_dir: str = 'results') -> pd.DataFrame:

    results = []
    
    # Verifica se o diretório existe
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Diretório {results_dir} não encontrado")
    
    # Carrega cada arquivo de resultado
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                result = json.load(f)
                results.append(result)
    
    # Converte para DataFrame
    df = pd.DataFrame(results)
    return df

def select_best_architecture(results_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Seleciona a melhor arquitetura com base no score OACE.
    
    Args:
        results_data: DataFrame com os resultados das arquiteturas
        
    Returns:
        Tuple contendo:
        - Dict com a configuração da melhor arquitetura
        - Dict com as métricas da melhor arquitetura
    """
    # Calcula pesos usando AHP
    wa = calculate_ahp_weights(dataset_a, weight_derivation)
    wc = calculate_ahp_weights(dataset_c, weight_derivation)
    
    # Mapeia os pesos para as métricas
    assertiveness_weights = {'precision': wa[0], 'accuracy': wa[1], 'recall': wa[2]}
    
    cost_weights = {'total_params': wc[0], 'avg_inference_time': wc[1], 'memory_used_mb': wc[2]}
    
    # Calcula limites min/max para normalização
    assertiveness_min_max = {
        'accuracy': {
            'min': results_data['accuracy'].min(),
            'max': results_data['accuracy'].max()
        },
        'precision': {
            'min': results_data['precision'].min(),
            'max': results_data['precision'].max()
        },
        'recall': {
            'min': results_data['recall'].min(),
            'max': results_data['recall'].max()
        }
    }
    
    print("assertiveness_min_max", assertiveness_min_max)
    
    cost_min_max = {
        'total_params': {
            'min': results_data['total_params'].min(),
            'max': results_data['total_params'].max()
        },
        'avg_inference_time': {
            'min': results_data['avg_inference_time'].min(),
            'max': results_data['avg_inference_time'].max()
        },
        'memory_used_mb': {
            'min': results_data['memory_used_mb'].min(),
            'max': results_data['memory_used_mb'].max()
        }
    }
    
    print("cost_min_max", cost_min_max)
    
    # Calcula score OACE para cada arquitetura
    oace_scores = []
    for _, row in results_data.iterrows():
        # Métricas de assertividade
        assertiveness_metrics = {
            'accuracy': row['accuracy'],
            'precision': row['precision'],
            'recall': row['recall']
        }
        print("row:", row)
        print("assertiveness_metrics:", assertiveness_metrics)
        
        # Métricas de custo
        cost_metrics = {
            'total_params': row['total_params'],
            'avg_inference_time': row['avg_inference_time'],
            'memory_used_mb': row['memory_used_mb']
        }
        
        print("cost_metrics:", cost_metrics)
        print("assertiveness_min_max: ", assertiveness_min_max)
        print("cost_min_max:", cost_min_max)
        
        # Calcula score OACE (lambda = 0.5 para balancear assertividade e custo)
        score, a_m, c_m = calculate_oace_score(
            assertiveness_metrics=assertiveness_metrics,
            cost_metrics=cost_metrics,
            lambda_param=0.5,
            assertiveness_weights=assertiveness_weights,
            cost_weights=cost_weights,
            assertiveness_min_max=assertiveness_min_max,
            cost_min_max=cost_min_max
        )
        
        oace_scores.append({
            'score': score,
            'a_m': a_m,
            'c_m': c_m
        })
        
        print("oace_scores", oace_scores)
        
    # Adiciona scores ao DataFrame
    results_data['oace_score'] = [s['score'] for s in oace_scores]
    results_data['assertiveness_score'] = [s['a_m'] for s in oace_scores]
    results_data['cost_score'] = [s['c_m'] for s in oace_scores]
    results_data.to_csv('results_data.csv', index=False)
    
    print("results_data", results_data)
    
    # Encontra a melhor arquitetura
    best_idx = results_data['oace_score'].idxmax()
    best_architecture = results_data.iloc[best_idx]
    
    # Prepara retorno
    best_config = {
        'model_type': best_architecture['model'],
        'architecture_config': best_architecture.get('architecture_config', {})
    }
    
    best_metrics = {
        'accuracy': float(best_architecture['accuracy']),
        'precision': float(best_architecture['precision']),
        'recall': float(best_architecture['recall']),
        'total_params': float(best_architecture['total_params']),
        'avg_inference_time': float(best_architecture['avg_inference_time']),
        'memory_used_mb': float(best_architecture['memory_used_mb']),
        'gflops': float(best_architecture['gflops']),
        'oace_score': float(best_architecture['oace_score']),
        'assertiveness_score': float(best_architecture['assertiveness_score']),
        'cost_score': float(best_architecture['cost_score'])
    }
    
    best_result = pd.DataFrame([best_config, best_metrics])
    best_result.to_csv('best_result.csv', index=False)
    
    # Imprime resultados
    print("\n=== Melhor Arquitetura Selecionada ===")
    print(f"Modelo: {best_config['model_type']}")
    
    print("\nPesos AHP Utilizados:")
    print("Assertividade:")
    for metric, weight in assertiveness_weights.items():
        print(f"  {metric}: {weight:.4f}")
    print("Custo:")
    for metric, weight in cost_weights.items():
        print(f"  {metric}: {weight:.4f}")
    
    print("\nMétricas de Assertividade:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"Score de Assertividade (A(m)): {best_metrics['assertiveness_score']:.4f}")
    
    print("\nMétricas de Custo:")
    print(f"Total de Parâmetros: {best_metrics['total_params']:.2e}")
    print(f"Tempo Médio de Inferência: {best_metrics['avg_inference_time']:.4f}s")
    print(f"Memória Utilizada: {best_metrics['memory_used_mb']:.2f}MB")
    print(f"GFLOPs: {best_metrics['gflops']:.4f}")
    print(f"Score de Custo (C(m)): {best_metrics['cost_score']:.4f}")
    
    print(f"\nScore OACE Final: {best_metrics['oace_score']:.4f}")
    
    return best_config, best_metrics

if __name__ == '__main__':
    # Exemplo de uso
    try:
        results_df = load_results_data()
        best_config, best_metrics = select_best_architecture(results_df)
    except Exception as e:
        print(f"Erro ao selecionar melhor arquitetura: {str(e)}") 