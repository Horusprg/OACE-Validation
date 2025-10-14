"""
Implementação corrigida do PSO que sempre inicializa o enxame manualmente.
"""

import numpy as np
import json
import os
from typing import Callable, Dict, Any, Optional, Tuple

try:
    import pyswarms as ps
except ImportError:
    print("⚠️ PySwarms não encontrado, usando implementação manual")
    ps = None

try:
    from optimizers.afsa import AFSA
except ImportError:
    print("⚠️ AFSA não encontrado, usando inicialização aleatória")
    AFSA = None

class PSO:
    """
    Implementação do Particle Swarm Optimization (PSO) com logging integrado.
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

        # Cria um objeto swarm manual
        class ManualSwarm:
            def __init__(self):
                self.position = None
                self.velocity = None
                self.pbest_pos = None
                self.pbest_cost = None
                self.best_pos = None
                self.best_cost = None
        
        class ManualOptimizer:
            def __init__(self):
                self.swarm = ManualSwarm()
                self.options = pso_options
        
        self.optimizer = ManualOptimizer()

    def fitness_function(self, x):
        """
        Função de aptidão (fitness). Para este exemplo, usamos a função esfera.
        """
        return np.sum(x**2, axis=1)

    def optimize(self, fitness_function=None, metrics_function=None, start_iteration=0):
        """
        Executa o processo de otimização usando PSO.
        """
        if fitness_function is not None:
            self.fitness_function = fitness_function

        # Sempre inicializa o enxame manualmente
        print("🔄 Inicializando enxame...")
        try:
            # Inicialização manual
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
            raise ValueError(f"Falha na inicialização do enxame: {e}")

        # Executa as iterações
        for i in range(start_iteration, self.max_iter):
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
                except Exception as e:
                    print(f"⚠️ Erro ao calcular métricas: {e}")

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
        # Calcula fitness atual
        fitness_values = self.fitness_function(self.optimizer.swarm.position)
        
        # Atualiza pbest se necessário
        for i in range(self.population_size):
            if fitness_values[i] < self.optimizer.swarm.pbest_cost[i]:
                self.optimizer.swarm.pbest_pos[i] = self.optimizer.swarm.position[i].copy()
                self.optimizer.swarm.pbest_cost[i] = fitness_values[i]
        
        # Atualiza gbest
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