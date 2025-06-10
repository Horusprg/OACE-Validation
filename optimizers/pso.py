import numpy as np
import pyswarms as ps
from .afsa import AFSA

class PSO:
    """
    Implementa o algoritmo Particle Swarm Optimization (PSO) usando PySwarms
    com otimização inicial pelo Artificial Fish Swarm Algorithm (AFSA).

    Esta implementação utiliza a biblioteca PySwarms para o PSO, mantendo
    a otimização inicial com AFSA para melhorar a convergência.

    Atributos:
        population_size (int): O número de partículas no enxame.
        n_dim (int): A dimensionalidade do espaço de busca.
        max_iter (int): O número máximo de iterações.
        lower_bound (float): O limite inferior do espaço de busca.
        upper_bound (float): O limite superior do espaço de busca.
        afsa_params (dict): Parâmetros para o AFSA.
        options (dict): Parâmetros para o PSO do PySwarms.
    """

    def __init__(self, population_size, n_dim, max_iter, lower_bound, upper_bound,
                 afsa_params=None, pso_options=None):
        self.population_size = population_size
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Parâmetros padrão para o AFSA
        if afsa_params is None:
            afsa_params = {
                'visual': 0.5,
                'step': 0.1,
                'try_times': 5,
                'max_iter': 50
            }
        self.afsa_params = afsa_params

        # Parâmetros padrão para o PSO
        if pso_options is None:
            pso_options = {
                'c1': 0.5,  # Coeficiente cognitivo
                'c2': 0.3,  # Coeficiente social
                'w': 0.9,   # Peso de inércia
                'k': 2,     # Número de vizinhos
                'p': 2      # Distância p-norma
            }
        self.options = pso_options

        # Inicializa o otimizador PSO
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=self.population_size,
            dimensions=self.n_dim,
            options=self.options,
            bounds=([self.lower_bound] * self.n_dim, 
                   [self.upper_bound] * self.n_dim)
        )

    def fitness_function(self, x):
        """
        Função de aptidão (fitness). Para este exemplo, usamos a função esfera.
        Em um cenário real, seria substituída pela função da fórmula (1) do artigo.

        Args:
            x (np.ndarray): Array de posições das partículas.

        Returns:
            np.ndarray: Valores de fitness para cada partícula.
        """
        return np.sum(x**2, axis=1)

    def initialize_with_afsa(self):
        """
        Usa o AFSA para otimizar as posições iniciais das partículas.

        Returns:
            np.ndarray: Posições otimizadas pelo AFSA.
        """
        afsa = AFSA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            visual=self.afsa_params['visual'],
            step=self.afsa_params['step'],
            try_times=self.afsa_params['try_times'],
            max_iter=self.afsa_params['max_iter'],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound
        )
        
        return afsa.optimize()

    def optimize(self):
        """
        Executa o processo de otimização usando PSO com inicialização pelo AFSA.

        Returns:
            tuple: (melhor posição encontrada, melhor valor de fitness)
        """
        # Inicializa as partículas usando AFSA
        initial_positions = self.initialize_with_afsa()
        
        # Configura as posições iniciais no otimizador
        self.optimizer.swarm.position = initial_positions
        
        # Executa a otimização
        cost, pos = self.optimizer.optimize(
            self.fitness_function,
            iters=self.max_iter,
            verbose=True
        )
        
        return pos, cost

# Exemplo de uso:
if __name__ == "__main__":
    # Criar instância do PSO
    pso = PSO(
        population_size=30,
        n_dim=2,
        max_iter=100,
        lower_bound=-10,
        upper_bound=10
    )
    
    # Executar otimização
    best_position, best_fitness = pso.optimize()
    
    print(f"Melhor posição encontrada: {best_position}")
    print(f"Melhor valor de fitness: {best_fitness}")