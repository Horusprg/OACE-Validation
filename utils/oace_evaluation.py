import numpy as np
import pandas as pd
from typing import Dict, Union

def calculate_oace_score(
    assertiveness_metrics: Dict[str, Union[int, float]],
    cost_metrics: Dict[str, Union[int, float]],
    lambda_param: float,
    assertiveness_weights: Dict[str, float],
    cost_weights: Dict[str, float],
    assertiveness_min_max: Dict[str, Dict[str, Union[int, float]]],
    cost_min_max: Dict[str, Dict[str, Union[int, float]]]
) -> float:
    """
    Calcula o score OACE para uma arquitetura de modelo de ML de forma otimizada usando pandas.

    Args:
        assertiveness_metrics (Dict): Valores brutos das métricas de assertividade.
        cost_metrics (Dict): Valores brutos das métricas de custo.
        lambda_param (float): Parâmetro de trade-off λ (entre 0 e 1).
        assertiveness_weights (Dict): Pesos para as métricas de assertividade.
        cost_weights (Dict): Pesos para as métricas de custo.
        assertiveness_min_max (Dict): Valores 'min' e 'max' para normalização da assertividade.
        cost_min_max (Dict): Valores 'min' e 'max' para normalização do custo.

    Returns:
        float: O score final Sϕ(m) entre 0 e 1.
    """
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("O parâmetro lambda_param deve estar no intervalo [0, 1].")
    print("Calculando score OACE...")
    # --- 1. Preparação dos Dados com Pandas ---
    # Converte os dicionários em pandas Series para facilitar os cálculos vetorizados.
    s_metrics = pd.Series(assertiveness_metrics)
    s_weights = pd.Series(assertiveness_weights)
    s_min = pd.Series({k: v['min'] for k, v in assertiveness_min_max.items()})
    s_max = pd.Series({k: v['max'] for k, v in assertiveness_min_max.items()})
    
    print(s_metrics)
    print(s_weights)
    print(s_min)
    print(s_max)

    c_metrics = pd.Series(cost_metrics)
    c_weights = pd.Series(cost_weights)
    c_min = pd.Series({k: v['min'] for k, v in cost_min_max.items()})
    c_max = pd.Series({k: v['max'] for k, v in cost_min_max.items()})

    print(c_metrics)
    print(c_weights)
    print(c_min)
    print(c_max)

    # --- 2. Normalização Vetorizada ---
    # Evita divisão por zero. Onde max == min, a normalização é 1 se o valor for >= min, senão 0.
    s_range = s_max - s_min
    c_range = c_max - c_min
    
    print(s_range)
    print(c_range)

    # A normalização é feita de uma só vez para todas as métricas.
    norm_s = np.where(s_range == 0, np.where(s_metrics >= s_min, 1.0, 0.0), (s_metrics - s_min) / s_range)
    norm_c = np.where(c_range == 0, 0.0, (c_metrics - c_min) / c_range)
    
    print(norm_s)
    print(norm_c)

    # --- 3. Cálculo Agregado Vetorizado ---
    # Multiplicação elemento a elemento e soma, tudo em uma única operação.
    a_m = np.sum(norm_s * s_weights)
    c_m_normalized = np.sum((1.0 - norm_c) * c_weights)

    print(a_m)
    print(c_m_normalized)

    # --- 4. Cálculo do Score Final Sϕ(m) ---
    s_phi_score = (lambda_param * a_m) + ((1 - lambda_param) * c_m_normalized)

    print(s_phi_score)

    return s_phi_score


# --- Exemplo de Uso ---
if __name__ == '__main__':
    print("🧪 Executando exemplo de teste para a função OACE...")

    # Pesos definidos pelo AHP (exemplo do artigo) 
    assertiveness_weights = {'accuracy': 0.188, 'precision': 0.731, 'recall': 0.081}
    cost_weights = {'MTP': 0.731, 'TPI': 0.188, 'MS': 0.081}

    # Limites (min/max) para normalização, que seriam coletados durante o processo de busca
    assertiveness_ranges = {
        'accuracy': {'min': 0.60, 'max': 0.95},
        'precision': {'min': 0.55, 'max': 0.98},
        'recall': {'min': 0.62, 'max': 0.96}
    }
    cost_ranges = {
        'MTP': {'min': 4_000_000, 'max': 86_000_000}, 
        'TPI': {'min': 0.015, 'max': 0.279},          
        'MS': {'min': 7, 'max': 330}                 
    }

    # --- Arquitetura Candidata 1: Modelo Robusto (Ex: InceptionV3 ou ResNet-50) ---
    robust_model_assertiveness = {'accuracy': 0.92, 'precision': 0.95, 'recall': 0.91}
    robust_model_cost = {'MTP': 26_000_000, 'TPI': 0.115, 'MS': 90}

    # --- Arquitetura Candidata 2: Modelo Leve (Ex: MobileNetV2) ---
    light_model_assertiveness = {'accuracy': 0.85, 'precision': 0.87, 'recall': 0.84}
    light_model_cost = {'MTP': 4_100_000, 'TPI': 0.017, 'MS': 8}

    print("\n" + "="*50)
    print("📊 Avaliando Cenários com a Função OACE")
    print("="*50)

    # Cenário 1: Priorizando Assertividade (λ = 0.5) 
    lambda_assertiveness = 0.5
    score_robust_assertiveness = calculate_oace_score(
        robust_model_assertiveness, robust_model_cost, lambda_assertiveness,
        assertiveness_weights, cost_weights, assertiveness_ranges, cost_ranges
    )
    score_light_assertiveness = calculate_oace_score(
        light_model_assertiveness, light_model_cost, lambda_assertiveness,
        assertiveness_weights, cost_weights, assertiveness_ranges, cost_ranges
    )
    
    print(f"Modelo Robusto:")
    print(f"Sϕ(m) = {score_robust_assertiveness}")
    
    print(f"Modelo Leve:")
    print(f"Sϕ(m) = {score_light_assertiveness}")
    