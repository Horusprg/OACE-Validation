import numpy as np
import pyswarms as ps
from .afsa import AFSA
import json
import os

class PSO:
    """
    Implementa o algoritmo Particle Swarm Optimization (PSO) usando PySwarms
    com otimização inicial pelo Artificial Fish Swarm Algorithm (AFSA).
    Agora com logging detalhado de pbest, gbest e checkpoints.

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
        logger (Logger): Objeto para logging detalhado.
    """

    def __init__(self, population_size, n_dim, max_iter, lower_bound, upper_bound,
                 afsa_params=None, pso_options=None, logger=None):
        self.population_size = population_size
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.logger = logger

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

    def save_state(self, filepath):
        """Salva o estado atual do PSO em um arquivo JSON."""
        state = {
            'position': self.optimizer.swarm.position.tolist(),
            'velocity': self.optimizer.swarm.velocity.tolist(),
            'pbest_pos': self.optimizer.swarm.pbest_pos.tolist(),
            'pbest_cost': self.optimizer.swarm.pbest_cost.tolist(),
            'gbest_pos': self.optimizer.swarm.best_pos.tolist(),
            'gbest_cost': float(self.optimizer.swarm.best_cost)
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath):
        """Carrega o estado do PSO de um arquivo JSON."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.optimizer.swarm.position = np.array(state['position'])
        self.optimizer.swarm.velocity = np.array(state['velocity'])
        self.optimizer.swarm.pbest_pos = np.array(state['pbest_pos'])
        self.optimizer.swarm.pbest_cost = np.array(state['pbest_cost'])
        self.optimizer.swarm.best_pos = np.array(state['gbest_pos'])
        self.optimizer.swarm.best_cost = state['gbest_cost']

    def optimize(self, fitness_function=None, metrics_function=None):
        """
        Executa o processo de otimização usando PSO, registrando o histórico de pbest/gbest.
        Args:
            fitness_function: função de fitness customizada (opcional)
            metrics_function: função que recebe uma posição e retorna métricas detalhadas (opcional)
        Returns:
            tuple: (melhor posição encontrada, melhor valor de fitness)
        """
        if fitness_function is not None:
            self.fitness_function = fitness_function

        # Se as posições não foram inicializadas externamente, usa AFSA
        if self.optimizer.swarm.position is None:
            initial_positions = self.initialize_with_afsa()
            self.optimizer.swarm.position = initial_positions

        # Inicializa histórico
        for i in range(self.max_iter):
            # Executa uma iteração manual do PSO
            cost, _ = self.optimizer.step(self.fitness_function)

            # Coleta dados do enxame
            population = np.copy(self.optimizer.swarm.position)
            fitness_values = np.copy(self.fitness_function(population))
            pbest_pos = np.copy(self.optimizer.swarm.pbest_pos)
            pbest_cost = np.copy(self.optimizer.swarm.pbest_cost)
            gbest_pos = np.copy(self.optimizer.swarm.best_pos)
            gbest_cost = float(self.optimizer.swarm.best_cost)

            # Métricas detalhadas do gbest (se função fornecida)
            metrics = None
            if metrics_function is not None:
                metrics = metrics_function(gbest_pos)

            # Logging detalhado
            if self.logger is not None:
                self.logger.log_iteration(
                    iteration=i+1,
                    phase="PSO",
                    population=population,
                    fitness_values=fitness_values,
                    best_position=gbest_pos,
                    best_fitness=gbest_cost,
                    metrics=metrics,
                    pbest_pos=pbest_pos,
                    pbest_cost=pbest_cost,
                    gbest_pos=gbest_pos,
                    gbest_cost=gbest_cost
                )
                # Checkpoint a cada 10 iterações
                if (i+1) % 10 == 0:
                    checkpoint_path = os.path.join(self.logger.log_dir, self.logger.current_experiment, f"pso_checkpoint_{i+1}.json")
                    self.save_state(checkpoint_path)

        # Retorna o melhor encontrado
        return self.optimizer.swarm.best_pos, self.optimizer.swarm.best_cost

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