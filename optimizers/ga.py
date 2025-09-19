import numpy as np
from deap import base, creator, tools, algorithms

# Corrige a criação dos tipos do DEAP para evitar múltiplas criações
#if not hasattr(creator, "FitnessMin"):
#    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#if not hasattr(creator, "Individual"):
#    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # ✅ POSITIVO
if not hasattr(creator, "Individual"):
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)  # ✅ FitnessMax
    
class GA:
    """
    Implementa o Algoritmo Genético (GA) usando a biblioteca DEAP para otimização global.
    
    Este algoritmo utiliza operadores genéticos (crossover e mutação) para refinar
    a busca por soluções globais ótimas, conforme descrito no Algoritmo 2 do artigo.
    
    Atributos:
        population_size (int): Tamanho da população.
        n_dim (int): Dimensionalidade do espaço de busca.
        max_iter (int): Número máximo de iterações.
        lower_bound (float): Limite inferior do espaço de busca.
        upper_bound (float): Limite superior do espaço de busca.
        initial_crossover_rate (float): Taxa inicial de crossover.
        initial_mutation_rate (float): Taxa inicial de mutação.
        tournament_size (int): Tamanho do torneio para seleção.
    """

    def __init__(self, population_size, n_dim, max_iter, lower_bound, upper_bound,
                 initial_crossover_rate=0.8, initial_mutation_rate=0.1, tournament_size=3):
        self.population_size = population_size
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.initial_crossover_rate = initial_crossover_rate
        self.initial_mutation_rate = initial_mutation_rate
        self.tournament_size = tournament_size
        
        # Inicialização dos componentes do DEAP
        self._setup_deap()
        
        # Inicialização da população
        self.population = None
        self.best_solution = None
        self.best_fitness = float('-inf')  # ✅ Mudança para -inf (maximização)

    def _setup_deap(self):
        """
        Configura os componentes do DEAP (tipos, toolbox, etc.).
        """
        # Configurar o toolbox
        self.toolbox = base.Toolbox()
        
        # Registrar atributos
        self.toolbox.register("attr_float", np.random.uniform, 
                            self.lower_bound, self.upper_bound)
        
        # Registrar estrutura e inicialização
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=self.n_dim)
        self.toolbox.register("population", tools.initRepeat, list,
                            self.toolbox.individual)
        
        # Registrar operadores genéticos
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mate", self._crossover_operation)
        self.toolbox.register("mutate", self._mutation_operation)
        self.toolbox.register("select", tools.selTournament,
                            tournsize=self.tournament_size)

    def fitness_function(self, individual):
        """
        Função de aptidão (fitness). Para este exemplo, usamos a função esfera.
        Em um cenário real, seria substituída pela função da fórmula (1) do artigo.
        
        Args:
            individual (np.ndarray): Indivíduo a ser avaliado.
            
        Returns:
            tuple: Valor da função objetivo (DEAP requer uma tupla).
        """
        # Converte para numpy array se necessário
        if not isinstance(individual, np.ndarray):
            individual = np.array(individual)
        return (np.sum(individual**2),)

    def _crossover_operation(self, ind1, ind2):
        """
        Operação de crossover adaptativa (fórmula 18).
        
        Args:
            ind1 (np.ndarray): Primeiro pai.
            ind2 (np.ndarray): Segundo pai.
            
        Returns:
            tuple: Dois filhos gerados pelo crossover.
        """
        # Crossover aritmético
        alpha = np.random.random()
        child1 = alpha * ind1 + (1 - alpha) * ind2
        child2 = (1 - alpha) * ind1 + alpha * ind2
        
        # Garante que os valores permaneçam dentro dos limites
        child1 = np.clip(child1, self.lower_bound, self.upper_bound)
        child2 = np.clip(child2, self.lower_bound, self.upper_bound)
        
        # Converte para o tipo Individual
        child1 = creator.Individual(child1)
        child2 = creator.Individual(child2)
        
        return child1, child2

    def _mutation_operation(self, individual, indpb=None):
        """
        Operação de mutação gaussiana (fórmulas 19 e 20).
        
        Args:
            individual (np.ndarray): Indivíduo a ser mutado.
            indpb (float, optional): Probabilidade de mutação por gene. Se None, usa a taxa adaptativa.
            
        Returns:
            tuple: Indivíduo mutado (DEAP requer uma tupla).
        """
        if indpb is None:
            indpb = self.initial_mutation_rate  # Usa a taxa inicial como padrão

        for i in range(len(individual)):
            if np.random.random() < indpb:
                # Mutação gaussiana
                individual[i] += np.random.normal(0, 0.1)
                # Garante que o valor permaneça dentro dos limites
                individual[i] = np.clip(individual[i], 
                                      self.lower_bound, 
                                      self.upper_bound)
        return individual,

    def adaptive_crossover_rate(self, iter_num):
        """
        Calcula a taxa de crossover adaptativa (fórmula 16).
        
        Args:
            iter_num (int): Número da iteração atual.
            
        Returns:
            float: Taxa de crossover adaptativa.
        """
        return self.initial_crossover_rate * (1 - iter_num / self.max_iter)

    def adaptive_mutation_rate(self, iter_num):
        """
        Calcula a taxa de mutação adaptativa (fórmula 17).
        
        Args:
            iter_num (int): Número da iteração atual.
            
        Returns:
            float: Taxa de mutação adaptativa.
        """
        return self.initial_mutation_rate * (iter_num / self.max_iter)

    def initialize_population(self, initial_population):
        """
        Inicializa a população com as soluções do PSO.
        
        Args:
            initial_population (np.ndarray): População inicial do PSO.
        """
        print(f"🔍 DEBUG GA: Inicializando população com {len(initial_population)} soluções")
        print(f"🔍 DEBUG GA: Tipo de initial_population: {type(initial_population)}")
        
        self.population = []
        for i, solution in enumerate(initial_population):
            print(f"🔍 DEBUG GA: Processando solução {i}: {solution}")
            print(f"🔍 DEBUG GA: Tipo da solução: {type(solution)}")
            
            ind = creator.Individual(solution)
            print(f"🔍 DEBUG GA: Indivíduo criado: {ind}")
            print(f"🔍 DEBUG GA: Tipo do indivíduo: {type(ind)}")
            
            print(f"🔍 DEBUG GA: Chamando fitness_function para indivíduo {i}")
            fitness_result = self.fitness_function(ind)
            print(f"🔍 DEBUG GA: Resultado do fitness: {fitness_result}")
            print(f"🔍 DEBUG GA: Tipo do resultado: {type(fitness_result)}")
            
            ind.fitness.values = fitness_result
            print(f"🔍 DEBUG GA: Fitness atribuído: {ind.fitness.values}")
            
            self.population.append(ind)
            print(f"🔍 DEBUG GA: Indivíduo {i} adicionado à população")
        
        print(f"🔍 DEBUG GA: População criada com {len(self.population)} indivíduos")
        print(f"🔍 DEBUG GA: Primeiro indivíduo: {self.population[0]}")
        print(f"🔍 DEBUG GA: Fitness do primeiro: {self.population[0].fitness.values}")
        
        # Atualiza a melhor solução
        self._update_best_solution()
        print(f"🔍 DEBUG GA: Melhor solução atualizada: {self.best_fitness}")

    def _update_best_solution(self):
        """
        Atualiza a melhor solução encontrada.
        """
        best_ind = tools.selBest(self.population, k=1)[0]
        if best_ind.fitness.values[0] > self.best_fitness:
            self.best_fitness = best_ind.fitness.values[0]
            self.best_solution = np.copy(best_ind)

    def optimize(self):
        """
        Executa o processo de otimização do GA.
        Returns:
            tuple: (melhor posição encontrada, melhor valor de fitness)
        """
        print(f"🧬 Iniciando otimização GA com {self.max_iter} iterações...")
        print(f"   • População inicial: {len(self.population)} indivíduos")
        print(f"   • Dimensões: {len(self.population[0])}")
        print(f"🔍 DEBUG GA: Melhor fitness inicial: {self.best_fitness}")
        print(f"🔍 DEBUG GA: Melhor solução inicial: {self.best_solution}")
            
        for iter_num in range(self.max_iter):
            print(f"🔍 DEBUG GA: Iteração {iter_num + 1}/{self.max_iter}")
            
            # Atualiza as taxas adaptativas
            crossover_rate = self.adaptive_crossover_rate(iter_num)
            mutation_rate = self.adaptive_mutation_rate(iter_num)
            print(f"🔍 DEBUG GA: Taxas - Crossover: {crossover_rate:.3f}, Mutação: {mutation_rate:.3f}")
            
            # Seleciona a próxima geração
            print(f"🔍 DEBUG GA: Gerando offspring...")
            offspring = algorithms.varOr(self.population, self.toolbox,
                                       lambda_=self.population_size,
                                       cxpb=crossover_rate,
                                       mutpb=mutation_rate)
            print(f"🔍 DEBUG GA: Offspring gerado com {len(offspring)} indivíduos")
            
            # Avalia os indivíduos
            print(f"🔍 DEBUG GA: Avaliando offspring...")
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for i, (fit, ind) in enumerate(zip(fits, offspring)):
                print(f"🔍 DEBUG GA: Offspring {i} - Fitness antes: {ind.fitness.values}")
                ind.fitness.values = fit
                print(f"🔍 DEBUG GA: Offspring {i} - Fitness depois: {ind.fitness.values}")
            
            # Atualiza a população
            print(f"🔍 DEBUG GA: Atualizando população...")
            self.population = offspring
            
            # Atualiza a melhor solução
            print(f"🔍 DEBUG GA: Atualizando melhor solução...")
            self._update_best_solution()
            print(f"🔍 DEBUG GA: Melhor fitness após iteração {iter_num + 1}: {self.best_fitness}")
            
            # Elitismo: mantém a melhor solução
            if self.best_solution is not None:
                print(f"🔍 DEBUG GA: Aplicando elitismo...")
                worst_idx = np.argmax([ind.fitness.values[0] for ind in self.population])
                print(f"🔍 DEBUG GA: Substituindo indivíduo {worst_idx} (fitness: {self.population[worst_idx].fitness.values[0]:.6f})")
                self.population[worst_idx] = creator.Individual(self.best_solution)
                self.population[worst_idx].fitness.values = (self.best_fitness,)
                print(f"🔍 DEBUG GA: Indivíduo substituído com fitness: {self.population[worst_idx].fitness.values[0]:.6f}")
                
        print(f"\n✅ GA concluído!")
        print(f"   • Melhor fitness final: {self.best_fitness:.6f}")
        print(f"   • Melhor posição: {self.best_solution}")
        print(f"🔍 DEBUG GA: Tipo do best_fitness: {type(self.best_fitness)}")
        print(f"🔍 DEBUG GA: best_fitness é numpy array? {isinstance(self.best_fitness, np.ndarray)}")
        
        return self.best_solution, self.best_fitness

# Exemplo de uso:
if __name__ == "__main__":
    # Criar instância do GA
    ga = GA(
        population_size=30,
        n_dim=2,
        max_iter=100,
        lower_bound=-10,
        upper_bound=10
    )
    
    # Inicializar com uma população aleatória
    initial_population = np.random.uniform(-10, 10, (30, 2))
    ga.initialize_population(initial_population)
    
    # Executar otimização
    best_position, best_fitness = ga.optimize()
    
    print(f"Melhor posição encontrada: {best_position}")
    print(f"Melhor valor de fitness: {best_fitness}") 