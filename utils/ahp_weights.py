import numpy as np
from pyDecision.algorithm import ahp_method

metrics_a_labels = ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]
metrics_c_labels = ["total_params", "avg_inference_time", "memory_used_mb", "gflops"] 
weight_derivation = 'max_eigen' # 'mean'; 'geometric' or 'max_eigen'

def critical_scenario_weights():
  """
  Foco: Máxima Assertividade.
  """
  matrix_a = np.array([
    # --- CENÁRIO MÉDICO (Crítico) ---
    # Foco: Recall (FN) e Precision (FP) dominam. Top-5 é irrelevante.
    # Saaty Logic:
    # Recall é 9x mais importante que Top-5.
    # Precision é 7x mais importante que Top-1.
    # F1 é o balanço, 5x mais importante que Top-1.
    # T1   T5   Prec Rec  F1
    [1.0, 5.0, 1/5, 1/7, 1/5], # Top1 (Baixa prioridade vs Prec/Rec)
    [1/5, 1.0, 1/7, 1/9, 1/7], # Top5 (Irrelevante)
    [5.0, 7.0, 1.0, 1/3, 1.0], # Precision (Crítico para evitar FP)
    [7.0, 9.0, 3.0, 1.0, 3.0], # Recall (Máxima prioridade - evitar FN/Morte)
    [5.0, 7.0, 1.0, 1/3, 1.0]  # F1 (Média harmônica)
  ])

  # Custo: Secundário, mas Memória importa para imagens médicas grandes (3D DICOM etc)
  matrix_c = np.array([
      # Par  Time Mem  GFL
      [1.0, 1.0, 1/3, 1.0], # Params (Igual a Time/GFL, menor que Mem)
      [1.0, 1.0, 1/5, 1.0], # Time   (Igual a Par/GFL, menor que Mem)
      [3.0, 5.0, 1.0, 5.0], # Memory (Importante: Gargalo) -> Note que inverti os valores originais para manter a lógica
      [1.0, 1.0, 1/5, 1.0]  # GFLOPs (Igual a Par/Time, menor que Mem)
  ])
  # Cálculo AHP
  wa, rc_a = ahp_method(matrix_a, wd=weight_derivation)
  wc, rc_c = ahp_method(matrix_c, wd=weight_derivation)
  
  wa_dict = dict(zip(metrics_a_labels, wa))
  wc_dict = dict(zip(metrics_c_labels, wc))
    
  return wa_dict, wc_dict, rc_a, rc_c
  
def equilibrium_scenario_weights():
  """
  Retorna pesos para 'Balanced Resource Model'.
  Foco: Equilíbrio entre Precisão e Custo.
  """
  # --- ASSERTIVIDADE ---
  matrix_a = np.array([
      # T1   T5   Prec Rec  F1
      [1.0, 3.0, 1.0, 1.0, 1/3], # Top1 (Padrão de mercado)
      [1/3, 1.0, 1/3, 1/3, 1/5], # Top5 (Auxiliar)
      [1.0, 3.0, 1.0, 1.0, 1.0], # Precision
      [1.0, 3.0, 1.0, 1.0, 1.0], # Recall
      [3.0, 5.0, 1.0, 1.0, 1.0]  # F1 (O melhor balanceador)
  ])

  # Custo: Tempo e GFLOPs impactam experiência do usuário
  matrix_c = np.array([
      # Par  Time Mem  GFL
      [1.0, 1/5, 1.0, 1/3], # Params (Menor que Time e GFL)
      [5.0, 1.0, 5.0, 3.0], # Time   (Maior que todos)
      [1.0, 1/5, 1.0, 1/3], # Mem    (Igual Params)
      [3.0, 1/3, 3.0, 1.0]  # GFLOPs (Intermediário)
  ])

  # Cálculo AHP
  wa, rc_a = ahp_method(matrix_a, wd=weight_derivation)
  wc, rc_c = ahp_method(matrix_c, wd=weight_derivation)
  
  wa_dict = dict(zip(metrics_a_labels, wa))
  wc_dict = dict(zip(metrics_c_labels, wc))
    
  return wa_dict, wc_dict, rc_a, rc_c

def limited_scenario_weights():
  """
  Retorna pesos para 'Educational Web Game (Offline)'.
  Foco: Precisão Pedagógica (Zero Falsos Positivos) + Viabilidade Web (Tamanho/Memória).
   """
  # --- ASSERTIVIDADE ---
  matrix_a = np.array([
      # T1   T5   Prec Rec  F1
      [1.0, 7.0, 1/3, 5.0, 3.0], # Top1 (Fundamental para o feedback do jogo)
      [1/7, 1.0, 1/9, 1/3, 1/5], # Top5 (Inútil para crianças)
      [3.0, 9.0, 1.0, 7.0, 5.0], # Precision (Crítico: não ensinar errado)
      [1/5, 3.0, 1/7, 1.0, 1/3], # Recall (Menos grave, criança pode tentar de novo)
      [1/3, 5.0, 1/5, 3.0, 1.0]  # F1 (Média)
  ])
  
  # --- CUSTO ---
  matrix_c = np.array([
      # Par  Time Mem  GFL
      [1.0, 3.0, 5.0, 7.0], # Params (Maior que Time, Mem, GFL)
      [1/3, 1.0, 3.0, 5.0], # Time   (Menor que Par, maior que Mem/GFL)
      [1/5, 1/3, 1.0, 3.0], # Mem    (Menor que Par/Time, maior que GFL)
      [1/7, 1/5, 1/3, 1.0]  # GFLOPs (Menor de todos)
  ])

  # Cálculo AHP
  wa, rc_a = ahp_method(matrix_a, wd=weight_derivation)
  wc, rc_c = ahp_method(matrix_c, wd=weight_derivation)
  
  wa_dict = dict(zip(metrics_a_labels, wa))
  wc_dict = dict(zip(metrics_c_labels, wc))
    
  return wa_dict, wc_dict, rc_a, rc_c

if __name__ == "__main__":
  
  wa_crit, wc_crit, rc_a_crit, rc_c_crit = critical_scenario_weights()
  wa_eq, wc_eq, rc_a_eq, rc_c_eq = equilibrium_scenario_weights()
  wa_lim, wc_lim, rc_a_lim, rc_c_lim = limited_scenario_weights()
  
  print("Critical Scenario Weights:")
  print("Assertiveness Weights:", wa_crit)
  print("Cost Weights:", wc_crit)
  print("Consistency Ratios:", rc_a_crit, rc_c_crit)
  print("\nEquilibrium Scenario Weights:")
  print("Assertiveness Weights:", wa_eq)
  print("Cost Weights:", wc_eq)
  print("Consistency Ratios:", rc_a_eq, rc_c_eq)
  print("\nLimited Scenario Weights:")
  print("Assertiveness Weights:", wa_lim)
  print("Cost Weights:", wc_lim)
  print("Consistency Ratios:", rc_a_lim, rc_c_lim)


