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

        # Inicializa o otimizador PSO (mas não inicializa o enxame ainda)
        try:
            self.optimizer = ps.single.GlobalBestPSO(
                n_particles=self.population_size,
                dimensions=self.n_dim,
                options=self.options,
                bounds=([self.lower_bound] * self.n_dim, 
                       [self.upper_bound] * self.n_dim)
            )
        except Exception as e:
            print(f"⚠️ Erro ao criar otimizador PySwarms: {e}")
            # Cria um objeto dummy se PySwarms falhar
            class DummySwarm:
                def __init__(self):
                    self.position = None
                    self.velocity = None
                    self.pbest_pos = None
                    self.pbest_cost = None
                    self.best_pos = None
                    self.best_cost = None
            
            class DummyOptimizer:
                def __init__(self):
                    self.swarm = DummySwarm()
                    self.options = pso_options
            
            self.optimizer = DummyOptimizer()

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
            'gbest_cost': float(self.optimizer.swarm.best_cost),
            'options': self.options,
            'afsa_params': self.afsa_params,
            'population_size': self.population_size,
            'n_dim': self.n_dim,
            'max_iter': self.max_iter,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath):
        """Carrega o estado do PSO de um arquivo JSON."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Carrega os dados do enxame
        self.optimizer.swarm.position = np.array(state['position'])
        self.optimizer.swarm.velocity = np.array(state['velocity'])
        self.optimizer.swarm.pbest_pos = np.array(state['pbest_pos'])
        self.optimizer.swarm.pbest_cost = np.array(state['pbest_cost'])
        self.optimizer.swarm.best_pos = np.array(state['gbest_pos'])
        self.optimizer.swarm.best_cost = state['gbest_cost']
        
        # Carrega parâmetros se disponíveis
        if 'options' in state:
            self.options = state['options']
        if 'afsa_params' in state:
            self.afsa_params = state['afsa_params']

    def load_state_from_dict(self, state_dict):
        """Carrega o estado do PSO a partir de um dicionário."""
        if 'position' in state_dict:
            self.optimizer.swarm.position = np.array(state_dict['position'])
        if 'velocity' in state_dict:
            self.optimizer.swarm.velocity = np.array(state_dict['velocity'])
        if 'pbest_pos' in state_dict:
            self.optimizer.swarm.pbest_pos = np.array(state_dict['pbest_pos'])
        if 'pbest_cost' in state_dict:
            self.optimizer.swarm.pbest_cost = np.array(state_dict['pbest_cost'])
        if 'gbest_pos' in state_dict:
            self.optimizer.swarm.best_pos = np.array(state_dict['gbest_pos'])
        if 'gbest_cost' in state_dict:
            self.optimizer.swarm.best_cost = state_dict['gbest_cost']

    def get_optimizer_state(self):
        """Retorna o estado atual do otimizador como dicionário."""
        return {
            'position': self.optimizer.swarm.position.tolist() if self.optimizer.swarm.position is not None else None,
            'velocity': self.optimizer.swarm.velocity.tolist() if self.optimizer.swarm.velocity is not None else None,
            'pbest_pos': self.optimizer.swarm.pbest_pos.tolist() if self.optimizer.swarm.pbest_pos is not None else None,
            'pbest_cost': self.optimizer.swarm.pbest_cost.tolist() if self.optimizer.swarm.pbest_cost is not None else None,
            'gbest_pos': self.optimizer.swarm.best_pos.tolist() if self.optimizer.swarm.best_pos is not None else None,
            'gbest_cost': float(self.optimizer.swarm.best_cost) if self.optimizer.swarm.best_cost is not None else None
        }

    def optimize(self, fitness_function=None, metrics_function=None, start_iteration=0):
        """
        Executa o processo de otimização usando PSO, registrando o histórico de pbest/gbest.
        
        Args:
            fitness_function: função de fitness customizada (opcional)
            metrics_function: função que recebe uma posição e retorna métricas detalhadas (opcional)
            start_iteration: iteração inicial (para retomar de checkpoint)
            
        Returns:
            tuple: (melhor posição encontrada, melhor valor de fitness)
        """
        if fitness_function is not None:
            self.fitness_function = fitness_function

        # Inicializa o enxame se necessário
        if self.optimizer.swarm.position is None:
            print("🔄 Inicializando enxame...")
            
            # Verifica se a função de fitness está definida
            if not hasattr(self, 'fitness_function') or self.fitness_function is None:
                raise ValueError("Função de fitness não está definida")
            
            try:
                # Tenta usar AFSA se disponível
                if hasattr(self, 'afsa_params') and self.afsa_params is not None:
                    initial_positions = self.initialize_with_afsa()
                else:
                    # Inicialização manual se AFSA não estiver disponível
                    initial_positions = np.random.uniform(
                        self.lower_bound, 
                        self.upper_bound, 
                        (self.population_size, self.n_dim)
                    )
                
                self.optimizer.swarm.position = initial_positions
                
                # Inicializa velocidade
                velocity_range = self.upper_bound - self.lower_bound
                self.optimizer.swarm.velocity = np.random.uniform(
                    -velocity_range * 0.1,
                    velocity_range * 0.1,
                    (self.population_size, self.n_dim)
                )
                
                # Inicializa pbest e gbest
                fitness_values = self.fitness_function(self.optimizer.swarm.position)
                self.optimizer.swarm.pbest_pos = self.optimizer.swarm.position.copy()
                self.optimizer.swarm.pbest_cost = fitness_values.copy()
                
                best_idx = np.argmin(self.optimizer.swarm.pbest_cost)
                self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[best_idx].copy()
                self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[best_idx]
                
                print(f"✅ Enxame inicializado com {self.population_size} partículas")
                
            except Exception as e:
                print(f"❌ Erro na inicialização: {e}")
                # Inicialização de emergência
                self.optimizer.swarm.position = np.random.uniform(
                    self.lower_bound, 
                    self.upper_bound, 
                    (self.population_size, self.n_dim)
                )
                velocity_range = self.upper_bound - self.lower_bound
                self.optimizer.swarm.velocity = np.random.uniform(
                    -velocity_range * 0.1,
                    velocity_range * 0.1,
                    (self.population_size, self.n_dim)
                )
                
                fitness_values = self.fitness_function(self.optimizer.swarm.position)
                self.optimizer.swarm.pbest_pos = self.optimizer.swarm.position.copy()
                self.optimizer.swarm.pbest_cost = fitness_values.copy()
                
                best_idx = np.argmin(self.optimizer.swarm.pbest_cost)
                self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[best_idx].copy()
                self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[best_idx]

        # Verifica se o enxame foi inicializado corretamente
        if (self.optimizer.swarm.position is None or 
            self.optimizer.swarm.position.size == 0 or
            self.optimizer.swarm.pbest_cost is None or
            self.optimizer.swarm.pbest_cost.size == 0):
            print(f"❌ Debug - position: {self.optimizer.swarm.position}")
            print(f"❌ Debug - position size: {self.optimizer.swarm.position.size if self.optimizer.swarm.position is not None else 'None'}")
            print(f"❌ Debug - pbest_cost: {self.optimizer.swarm.pbest_cost}")
            print(f"❌ Debug - pbest_cost size: {self.optimizer.swarm.pbest_cost.size if self.optimizer.swarm.pbest_cost is not None else 'None'}")
            raise ValueError("Enxame não foi inicializado corretamente")

        # Inicializa histórico
        for i in range(start_iteration, self.max_iter):
            # Executa uma iteração manual do PSO
            try:
                # Atualiza manualmente uma iteração
                self._update_swarm_one_iteration()
                    
            except Exception as e:
                print(f"⚠️ Erro na iteração {i+1}: {e}")
                # Fallback: simula uma iteração
                self._simulate_iteration()

            # Coleta dados do enxame
            population = np.copy(self.optimizer.swarm.position)
            fitness_values = np.copy(self.fitness_function(population))
            pbest_pos = np.copy(self.optimizer.swarm.pbest_pos)
            pbest_cost = np.copy(self.optimizer.swarm.pbest_cost)
            gbest_pos = np.copy(self.optimizer.swarm.best_pos)
            gbest_cost = float(self.optimizer.swarm.best_cost)

            # Métricas detalhadas do gbest (se função fornecida)
            metrics = None
            architecture_config = None
            if metrics_function is not None:
                try:
                    metrics = metrics_function(gbest_pos)
                    # Tenta extrair configuração da arquitetura se disponível
                    if hasattr(metrics_function, 'get_architecture_config'):
                        architecture_config = metrics_function.get_architecture_config(gbest_pos)
                except Exception as e:
                    print(f"⚠️ Erro ao calcular métricas: {e}")

            # Logging detalhado
            if self.logger is not None:
                # CORREÇÃO: Calcula o score OACE para o logging
                oace_score = None
                try:
                    if metrics is not None:
                        # Usa a função de fitness para calcular o score OACE
                        oace_score = self.fitness_function(gbest_pos)
                        # Se o fitness_function retorna um array, pega o primeiro valor
                        if isinstance(oace_score, np.ndarray):
                            oace_score = oace_score[0] if oace_score.size > 0 else None
                except Exception as e:
                    print(f"⚠️ Erro ao calcular score OACE: {e}")
                
                self.logger.log_iteration(
                    iteration=i+1,
                    phase="PSO",
                    population=population,
                    fitness_values=fitness_values,
                    best_position=gbest_pos,
                    best_fitness=gbest_cost,
                    metrics=metrics,
                    oace_score=oace_score,  # CORREÇÃO: Adiciona o score OACE
                    pbest_pos=pbest_pos,
                    pbest_cost=pbest_cost,
                    gbest_pos=gbest_pos,
                    gbest_cost=gbest_cost,
                    architecture_config=architecture_config
                )
                
                # Checkpoint a cada 10 iterações
                if (i+1) % 10 == 0:
                    optimizer_state = self.get_optimizer_state()
                    self.logger.log_checkpoint(
                        iteration=i+1,
                        phase="PSO",
                        population=population,
                        fitness_values=fitness_values,
                        best_position=gbest_pos,
                        best_fitness=gbest_cost,
                        optimizer_state=optimizer_state
                    )

        # Retorna o melhor encontrado
        return self.optimizer.swarm.best_pos, self.optimizer.swarm.best_cost

    def _update_swarm_one_iteration(self):
        """
        Atualiza o enxame por uma iteração usando a implementação manual do PSO.
        """
        # Verifica se o enxame está inicializado
        if (self.optimizer.swarm.position is None or 
            self.optimizer.swarm.position.size == 0):
            raise ValueError("Enxame não está inicializado")
        
        # Calcula fitness atual
        fitness_values = self.fitness_function(self.optimizer.swarm.position)
        
        # Verifica se fitness_values tem o tamanho correto
        if len(fitness_values) != self.population_size:
            raise ValueError(f"Fitness values tem tamanho {len(fitness_values)}, esperado {self.population_size}")
        
        # Atualiza pbest se necessário
        for i in range(self.population_size):
            if fitness_values[i] < self.optimizer.swarm.pbest_cost[i]:
                self.optimizer.swarm.pbest_pos[i] = self.optimizer.swarm.position[i].copy()
                self.optimizer.swarm.pbest_cost[i] = fitness_values[i]
        
        # Atualiza gbest
        if self.optimizer.swarm.pbest_cost.size > 0:
            best_idx = np.argmin(self.optimizer.swarm.pbest_cost)
            if self.optimizer.swarm.pbest_cost[best_idx] < self.optimizer.swarm.best_cost:
                self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[best_idx].copy()
                self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[best_idx]
        
        # Atualiza velocidade e posição
        w = self.options.get('w', 0.9)  # peso de inércia
        c1 = self.options.get('c1', 0.5)  # coeficiente cognitivo
        c2 = self.options.get('c2', 0.3)  # coeficiente social
        
        # Gera números aleatórios
        r1 = np.random.rand(self.population_size, self.n_dim)
        r2 = np.random.rand(self.population_size, self.n_dim)
        
        # Atualiza velocidade
        self.optimizer.swarm.velocity = (w * self.optimizer.swarm.velocity + 
                                        c1 * r1 * (self.optimizer.swarm.pbest_pos - self.optimizer.swarm.position) +
                                        c2 * r2 * (self.optimizer.swarm.best_pos - self.optimizer.swarm.position))
        
        # Atualiza posição
        self.optimizer.swarm.position += self.optimizer.swarm.velocity
        
        # Aplica limites
        self.optimizer.swarm.position = np.clip(
            self.optimizer.swarm.position, 
            self.lower_bound, 
            self.upper_bound
        )

    def _simulate_iteration(self):
        """
        Simula uma iteração quando há problemas com o otimizador.
        """
        # Simula movimento aleatório das partículas
        noise = np.random.normal(0, 0.1, self.optimizer.swarm.position.shape)
        self.optimizer.swarm.position += noise
        
        # Aplica limites
        self.optimizer.swarm.position = np.clip(
            self.optimizer.swarm.position, 
            self.lower_bound, 
            self.upper_bound
        )
        
        # Atualiza fitness
        fitness_values = self.fitness_function(self.optimizer.swarm.position)
        
        # Atualiza pbest e gbest
        for i in range(self.population_size):
            if fitness_values[i] < self.optimizer.swarm.pbest_cost[i]:
                self.optimizer.swarm.pbest_pos[i] = self.optimizer.swarm.position[i].copy()
                self.optimizer.swarm.pbest_cost[i] = fitness_values[i]
        
        best_idx = np.argmin(self.optimizer.swarm.pbest_cost)
        if self.optimizer.swarm.pbest_cost[best_idx] < self.optimizer.swarm.best_cost:
            self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[best_idx].copy()
            self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[best_idx]

    def resume_optimization(self, checkpoint_file, fitness_function=None, metrics_function=None):
        """
        Resume a otimização a partir de um checkpoint.
        
        Args:
            checkpoint_file: caminho para o arquivo de checkpoint
            fitness_function: função de fitness (opcional)
            metrics_function: função de métricas (opcional)
            
        Returns:
            tuple: (melhor posição encontrada, melhor valor de fitness)
        """
        # Carrega o checkpoint
        self.load_state(checkpoint_file)
        
        # Extrai informações do checkpoint
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        start_iteration = checkpoint_data.get('iteration', 0)
        
        print(f"🔄 Retomando otimização a partir da iteração {start_iteration}")
        
        # Continua a otimização
        return self.optimize(
            fitness_function=fitness_function,
            metrics_function=metrics_function,
            start_iteration=start_iteration
        )

    def get_optimization_info(self):
        """Retorna informações sobre o estado atual da otimização."""
        if self.optimizer.swarm.position is None:
            return {
                "status": "not_initialized",
                "best_fitness": None,
                "best_position": None,
                "population_size": self.population_size,
                "dimensions": self.n_dim
            }
        
        return {
            "status": "initialized",
            "best_fitness": float(self.optimizer.swarm.best_cost),
            "best_position": self.optimizer.swarm.best_pos.tolist(),
            "population_size": self.population_size,
            "dimensions": self.n_dim,
            "current_iteration": 0,  # Seria atualizado durante a otimização
            "max_iterations": self.max_iter
        }

    def initialize_swarm_with_population(self, initial_population):
        """
        Inicializa completamente o enxame do PSO com uma população específica.
        
        Args:
            initial_population (np.ndarray): População inicial para o enxame
        """
        if initial_population is None or initial_population.size == 0:
            raise ValueError("População inicial não pode ser None ou vazia")
        
        # Verifica se a população tem o formato correto
        if initial_population.ndim == 1:
            initial_population = initial_population.reshape(1, -1)
            print(f"initial_population in initialize_swarm_with_population(): {initial_population}")
        
        if initial_population.shape[1] != self.n_dim:
            raise ValueError(f"Dimensão da população ({initial_population.shape[1]}) não corresponde à dimensão esperada ({self.n_dim})")
        
        # Define a posição inicial
        self.optimizer.swarm.position = initial_population.copy()
        
        # Inicializa velocidade
        velocity_range = self.upper_bound - self.lower_bound
        self.optimizer.swarm.velocity = np.random.uniform(
            -velocity_range * 0.1,
            velocity_range * 0.1,
            initial_population.shape
        )
        
        # Calcula fitness inicial
        fitness_values = self.fitness_function(self.optimizer.swarm.position)
        print(f"fitness_values in initialize_swarm_with_population(): {fitness_values}")
        
        # Inicializa pbest
        self.optimizer.swarm.pbest_pos = self.optimizer.swarm.position.copy()
        self.optimizer.swarm.pbest_cost = fitness_values.copy()
        
        # Inicializa gbest
        best_idx = np.argmin(self.optimizer.swarm.pbest_cost)
        self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[best_idx].copy()
        self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[best_idx]
        
        print(f"self.optimizer.swarm.pbest_cost in initialize_swarm_with_population(): {self.optimizer.swarm.pbest_cost}")
        print(f"self.optimizer.swarm.best_cost in initialize_swarm_with_population(): {self.optimizer.swarm.best_cost}")
        
        print(f"✅ Enxame inicializado com {len(initial_population)} partículas")

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
    
    # Salvar checkpoint
    pso.save_state("pso_checkpoint.json")
    
    # Carregar e retomar
    pso2 = PSO(
        population_size=30,
        n_dim=2,
        max_iter=100,
        lower_bound=-10,
        upper_bound=10
    )
    
    best_position2, best_fitness2 = pso2.resume_optimization("pso_checkpoint.json")
    print(f"Retomada - Melhor posição: {best_position2}")
    print(f"Retomada - Melhor fitness: {best_fitness2}")