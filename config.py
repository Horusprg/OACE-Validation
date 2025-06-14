# =======================
# Parâmetros do PSO
# =======================
PSO_CONFIG = {
    "n_particles": 10,
    "max_iter": 50,
    "vmax": 100,
    "vmin": -100,
    "wmax": 0.9,
    "wmin": 0.4,
    "c1": 2.0,
    "c2": 2.0,
}
# =======================
# Parâmetros do AFSA
# =======================
AFSA_CONFIG = {
    "visual": 0.5,
    "step": 0.1,
    "try_times": 5,
    "max_iter": 50,
    "min_fitness": 0.01,
    "swarm_size": 10,
}
# =======================
# Parâmetros do GA
# =======================
GA_CONFIG = {
    "population_size": 10,
    "max_iter": 50,
    "pcmin": 0.6,   # probabilidade mínima de crossover
    "pcmax": 0.9,   # probabilidade máxima de crossover
    "pmmin": 0.01,  # probabilidade mínima de mutação
    "pmmax": 0.1,   # probabilidade máxima de mutação
    "tournament_size": 3,
}
# =======================
# Parâmetros do OACE
# =======================
OACE_CONFIG = {
    "lambda": 0.5,  # Peso de balanceamento entre assertividade e custo
}
# =======================
# Sistema para variação de parâmetros (exemplo de grid search)
# =======================
GRID_SEARCH = {
    "PSO": {
        "wmax": [0.9, 0.8],
        "wmin": [0.4, 0.3],
        "c1": [2.0, 2.5],
        "c2": [2.0, 2.5],
    },
    "OACE": {
        "lambda": [0.5]
    }
}
# =======================
# Função utilitária para acessar configs
# =======================
def get_config(alg_name):
    if alg_name == "PSO":
        return PSO_CONFIG
    elif alg_name == "AFSA":
        return AFSA_CONFIG
    elif alg_name == "GA":
        return GA_CONFIG
    elif alg_name == "OACE":
        return OACE_CONFIG
    else:
        raise ValueError("Algoritmo não reconhecido.")