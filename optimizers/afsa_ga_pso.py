import numpy as np
from .afsa import AFSA
from .pso import PSO
from .ga import GA

class AFSAGAPSO:
    """
    Implementa o algoritmo híbrido AFSA-GA-PSO para otimização de arquiteturas
    de redes neurais profundas, conforme descrito no Algoritmo 3 do artigo.
    
    Este algoritmo combina três técnicas de otimização:
    1. AFSA (Artificial Fish Swarm Algorithm) para otimização inicial
    2. PSO (Particle Swarm Optimization) para busca local
    3. GA (Genetic Algorithm) para refinamento global
    
    Atributos:
        population_size (int): Tamanho da população para cada algoritmo.
        n_dim (int): Dimensionalidade do espaço de busca (número de camadas/parâmetros).
        max_iter (int): Número máximo de iterações para cada algoritmo.
        lower_bound (float): Limite inferior do espaço de busca.
        upper_bound (float): Limite superior do espaço de busca.
        afsa_params (dict): Parâmetros para o AFSA.
        pso_params (dict): Parâmetros para o PSO.
        ga_params (dict): Parâmetros para o GA.
    """

    def __init__(self, population_size, n_dim, max_iter, lower_bound, upper_bound,
                 afsa_params=None, pso_params=None, ga_params=None):
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
        if pso_params is None:
            pso_params = {
                'v_max': 2.0,
                'v_min': -2.0,
                'w_max': 0.9,
                'w_min': 0.4,
                'c3': 1.5,
                'c4': 1.0
            }
        self.pso_params = pso_params
        
        # Parâmetros padrão para o GA
        if ga_params is None:
            ga_params = {
                'initial_crossover_rate': 0.8,
                'initial_mutation_rate': 0.1,
                'tournament_size': 3
            }
        self.ga_params = ga_params
        
        # Inicialização dos componentes
        self.afsa = None
        self.pso = None
        self.ga = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def fitness_function(self, x):
        """
        Função de aptidão (fitness) que avalia a qualidade da arquitetura.
        Esta função deve ser substituída pela função real de avaliação da rede.
        
        Args:
            x (np.ndarray): Arquitetura da rede (número de neurônios por camada).
            
        Returns:
            float: Valor da função objetivo (erro, acurácia, etc.).
        """
        # Exemplo: função esfera (substituir pelo OACE)
        return np.sum(x**2)

    def initialize_components(self):
        """
        Inicializa os componentes do algoritmo híbrido.
        """
        # Inicializa o AFSA
        self.afsa = AFSA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            visual=self.afsa_params['visual'],
            step=self.afsa_params['step'],
            try_times=self.afsa_params['try_times'],
            max_iter=self.afsa_params['max_iter'],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound
        )
        
        # Inicializa o PSO
        self.pso = PSO(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=self.max_iter,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            v_max=self.pso_params['v_max'],
            v_min=self.pso_params['v_min'],
            w_max=self.pso_params['w_max'],
            w_min=self.pso_params['w_min'],
            c3=self.pso_params['c3'],
            c4=self.pso_params['c4']
        )
        
        # Inicializa o GA
        self.ga = GA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=self.max_iter,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            initial_crossover_rate=self.ga_params['initial_crossover_rate'],
            initial_mutation_rate=self.ga_params['initial_mutation_rate'],
            tournament_size=self.ga_params['tournament_size']
        )

    def optimize(self):
        """
        Executa o processo de otimização híbrida conforme o Algoritmo 3.
        
        Returns:
            tuple: (melhor arquitetura encontrada, melhor valor de fitness)
        """
        # Passo 1: Inicializa os componentes
        self.initialize_components()
        
        # Passo 2: Executa o AFSA-PSO para obter temppbest
        print("Executando AFSA-PSO...")
        temppbest = self.afsa.optimize()
        
        # Passo 3: Resolve a aptidão do temppbest e inicializa swarm2
        print("Avaliando soluções iniciais...")
        initial_fitness = np.array([self.fitness_function(x) for x in temppbest])
        swarm2 = temppbest[initial_fitness.argsort()[:self.population_size]]
        
        # Passo 4: Executa o GA-PSO em swarm2
        print("Executando GA-PSO...")
        self.ga.initialize_population(swarm2)
        best_position, best_fitness = self.ga.optimize()
        
        # Passo 5: Registra a solução global ótima
        self.best_solution = best_position
        self.best_fitness = best_fitness
        
        # Passo 6: Seleciona a melhor estrutura
        print("Selecionando melhor estrutura...")
        selected_architecture = self.select_best_architecture(best_position)
        
        # Passo 7: Constrói o modelo otimizado
        print("Construindo modelo otimizado...")
        optimized_model = self.build_optimized_model(selected_architecture)
        
        return selected_architecture, self.best_fitness, optimized_model

    def select_best_architecture(self, best_position):
        """
        Seleciona a melhor estrutura de rede com base na solução otimizada.
        
        Args:
            best_position (np.ndarray): Melhor posição encontrada.
            
        Returns:
            dict: Arquitetura otimizada da rede.
        """
        # Arredonda os valores para obter números inteiros de neurônios
        architecture = np.round(best_position).astype(int)
        
        # Garante valores mínimos para cada camada
        architecture = np.maximum(architecture, 1)
        
        return {
            'hidden_layers': architecture.tolist(),
            'input_size': self.n_dim,
            'output_size': 1  # Ajustar conforme necessário
        }

    def build_optimized_model(self, architecture):
        """
        Constrói o modelo otimizado com a arquitetura selecionada.
        
        Args:
            architecture (dict): Arquitetura otimizada da rede.
            
        Returns:
            object: Modelo otimizado (implementar conforme framework usado).
        """
        # Implementar a construção do modelo com a arquitetura selecionada
        return {
            'architecture': architecture,
            'fitness': self.best_fitness,
            'parameters': {
                'afsa_params': self.afsa_params,
                'pso_params': self.pso_params,
                'ga_params': self.ga_params
            }
        }

# Exemplo de uso:
if __name__ == "__main__":
    # Criar instância do otimizador híbrido
    optimizer = AFSAGAPSO(
        population_size=30,
        n_dim=3,  # Número de camadas ocultas
        max_iter=100,
        lower_bound=1,    # Mínimo de neurônios por camada
        upper_bound=100   # Máximo de neurônios por camada
    )
    
    # Executar otimização
    best_architecture, best_fitness, model = optimizer.optimize()
    
    print(f"Melhor arquitetura encontrada: {best_architecture}")
    print(f"Melhor valor de fitness: {best_fitness}")
    print(f"Modelo otimizado: {model}") 