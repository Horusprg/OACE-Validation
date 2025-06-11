import json
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
    Calcula o score OACE para uma arquitetura de modelo de ML.

    Args:
        assertiveness_metrics (Dict): Dicionário com os valores brutos das métricas
            de assertividade (ex: {'accuracy': 0.85, 'precision': 0.88}).
        cost_metrics (Dict): Dicionário com os valores brutos das métricas de custo
            (ex: {'MTP': 25000000, 'TPI': 0.05}).
        lambda_param (float): O parâmetro λ (lambda), que controla o peso entre
            assertividade e custo. Varia de 0 a 1.
        assertiveness_weights (Dict): Pesos para cada métrica de assertividade,
            geralmente definidos pelo método AHP.
        cost_weights (Dict): Pesos para cada métrica de custo, definidos pelo AHP.
        assertiveness_min_max (Dict): Dicionário contendo os valores 'min' e 'max'
            para cada métrica de assertividade, necessários para a normalização.
        cost_min_max (Dict): Dicionário contendo os valores 'min' e 'max'
            para cada métrica de custo, necessários para a normalização.

    Returns:
        float: O score final Sϕ(m), um valor entre 0 e 1, onde valores mais
               altos indicam uma melhor avaliação geral da arquitetura.
    """
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("O parâmetro lambda_param deve estar no intervalo [0, 1].")

    # 1. Normalização das Métricas ---
    normalized_assertiveness = {}
    for metric, value in assertiveness_metrics.items():
        min_val = assertiveness_min_max[metric]['min']
        max_val = assertiveness_min_max[metric]['max']
        if max_val == min_val:
            normalized_assertiveness[metric] = 1.0 if value >= min_val else 0.0
        else:
            normalized_assertiveness[metric] = (value - min_val) / (max_val - min_val)

    normalized_cost = {}
    for metric, value in cost_metrics.items():
        min_val = cost_min_max[metric]['min']
        max_val = cost_min_max[metric]['max']
        if max_val == min_val:
            normalized_cost[metric] = 0.0
        else:
            normalized_cost[metric] = (value - min_val) / (max_val - min_val)

    # 2. Cálculo da Função de Assertividade Agregada A(m) 
    a_m = sum(normalized_assertiveness[m] * assertiveness_weights[m] for m in assertiveness_metrics)

    # 3. Cálculo da Função de Custo Agregada C(m) ---
    c_m_normalized = sum((1.0 - normalized_cost[m]) * cost_weights[m] for m in cost_metrics)

    # --- 4. Cálculo do Score Final Sϕ(m) ---
    s_phi_score = (lambda_param * a_m) + ((1 - lambda_param) * c_m_normalized)

    return s_phi_score, a_m, c_m_normalized

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
    score_robust_assertiveness, a_m, c_m_normalized = calculate_oace_score(
        robust_model_assertiveness, robust_model_cost, lambda_assertiveness,
        assertiveness_weights, cost_weights, assertiveness_ranges, cost_ranges
    )
    score_light_assertiveness, a_m_light, c_m_normalized_light = calculate_oace_score(
        light_model_assertiveness, light_model_cost, lambda_assertiveness,
        assertiveness_weights, cost_weights, assertiveness_ranges, cost_ranges
    )
    
    print(f"Modelo Robusto:")
    print(f"A(m) = {a_m}")
    print(f"C(m) = {c_m_normalized}")
    print(f"Sϕ(m) = {score_robust_assertiveness}")
    
    print(f"Modelo Leve:")
    print(f"A(m) = {a_m_light}")
    print(f"C(m) = {c_m_normalized_light}")
    print(f"Sϕ(m) = {score_light_assertiveness}")
    