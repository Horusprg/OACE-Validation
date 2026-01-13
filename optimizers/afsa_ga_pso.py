# source oace/bin/activate
# nohup python3 -X utf8 -u -m optimizers.afsa_ga_pso > results_3.log 2>&1 &
# python3 -X utf8 -u -m optimizers.afsa_ga_pso 2>&1 | tee teste5.log
import numpy as np
from optimizers.afsa import AFSA
from optimizers.pso import PSO
from optimizers.ga import GA
from models.architecture_loader import archictectures
from utils.oace_evaluation import calculate_oace_score
from typing import Dict, Any, Tuple, List, Type
from pydantic import BaseModel
import torch
from tqdm import tqdm
import sys
import os   
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import get_cifar10_dataloaders, get_wildshapes_dataloaders
from utils.optimization_logger import OptimizationLogger
from utils.ahp_weights import critical_scenario_weights, equilibrium_scenario_weights, limited_scenario_weights
import time

class AFSAGAPSO:
    """
    Implementa o algoritmo híbrido AFSA-GA-PSO para otimização de arquiteturas
    de redes neurais profundas usando o score OACE como função de fitness.
    Otimiza tanto a escolha da arquitetura quanto seus parâmetros.
    """

    def __init__(
        self,
        population_size: int,
        max_iter: int,
        train_loader,
        val_loader,
        test_loader,
        classes: List[str],
        lambda_param: float = 0.5,
        afsa_params: Dict[str, Any] = None,
        pso_params: Dict[str, Any] = None,
        ga_params: Dict[str, Any] = None,
        architectures_to_optimize: List[str] = None,
        log_dir: str = "results",
        device: torch.device = None
    ):
        # Arquiteturas disponíveis para otimização
        if architectures_to_optimize is None:
            self.architectures_to_optimize = list(archictectures.keys())
        else:
            self.architectures_to_optimize = architectures_to_optimize

        print(f"📋 Arquiteturas para otimização: {self.architectures_to_optimize}")

        # Arquiteturas e informações
        self.all_architectures = {
            name: archictectures[name] for name in self.architectures_to_optimize
        }
        self.population_size = population_size
        self.max_iter = max_iter
        self.lambda_param = lambda_param

        # DataLoaders e classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes
        
        # Configuração do dispositivo
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"🔧 Dispositivo configurado: {self.device}")

        # limites do espaço de busca
        self.param_bounds = self._get_unified_param_bounds()
        self.n_dim = len(self.param_bounds) + 1
        
        print(f"🎯 Dimensões do espaço de busca: {self.n_dim}")
        print(f"📏 Limites dos parâmetros: {self.param_bounds}")
        
        # Parâmetros padrão para o AFSA
        if afsa_params is None:
            afsa_params = {"visual": 0.5, "step": 0.1, "try_times": 5, "max_iter": 50}
        self.afsa_params = afsa_params
        
        # Parâmetros para o PSO 
        if pso_params is None:
            pso_params = {
                "c1": 1.5,    
                "c2": 1.5,    
                "w": 0.7,     
                "k": 3, 
                "p": 2
            }
        self.pso_params = pso_params
        
        # Parâmetros para o GA
        if ga_params is None:
            ga_params = {
                "initial_crossover_rate": 0.8,    
                "initial_mutation_rate": 0.15,   
                "tournament_size": 3,             
                "max_iter": 6
            }
        self.ga_params = ga_params
        
        # Inicialização dos componentes
        self.afsa = None
        self.pso = None
        self.ga = None
        self.best_solution = None
        self.best_fitness = float("-inf") 
        self.history = []

        # Métricas e limites
        self.metrics_history = []
        self.metrics_ranges = None
        
        # Cache para evitar re-avaliação de candidatos idênticos
        self.candidates_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Inicializa o logger
        self.logger = OptimizationLogger(log_dir=log_dir)
        
        from utils.log_prints import (
            _print_header, _print_section, _print_step, _print_configuration,
            _print_iteration_header, _print_candidate_details, _print_population_summary,
            _print_cache_stats, _print_phase_summary, _print_final_results
        )
        self._print_header = lambda title, width=80: _print_header(self, title, width)
        self._print_section = lambda title, width=60: _print_section(self, title, width)
        self._print_step = lambda step, details="": _print_step(self, step, details)
        self._print_configuration = lambda: _print_configuration(self)
        self._print_iteration_header = lambda phase, iteration, total_iterations=None: _print_iteration_header(self, phase, iteration, total_iterations)
        self._print_candidate_details = lambda candidate, metrics, architecture_name, architecture_params, oace_score=None, candidate_id=None: _print_candidate_details(self, candidate, metrics, architecture_name, architecture_params, oace_score, candidate_id)
        self._print_population_summary = lambda population, fitness_values, phase, iteration=None: _print_population_summary(self, population, fitness_values, phase, iteration)
        self._print_cache_stats = lambda: _print_cache_stats(self)
        self._print_phase_summary = lambda phase, best_fitness, best_architecture, best_params, total_time=None: _print_phase_summary(self, phase, best_fitness, best_architecture, best_params, total_time)
        self._print_final_results = lambda best_architecture, best_params, best_fitness, final_metrics: _print_final_results(self, best_architecture, best_params, best_fitness, final_metrics)
        
        experiment_config = {
            "population_size": population_size,
            "max_iter": max_iter,
            "lambda_param": lambda_param,
            "afsa_params": afsa_params,
            "pso_params": pso_params,
            "ga_params": ga_params,
            "architectures_to_optimize": architectures_to_optimize
        }
        self.logger.start_experiment(experiment_config)

    def _get_unified_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define os limites unificados do espaço de busca para todos os parâmetros
        de todas as arquiteturas disponíveis.
        """
        bounds = {}
        all_param_names = set()

        # Coleta parâmetros de todas as arquiteturas
        for arch_name, arch_info in self.all_architectures.items():
            params_class = type(arch_info["params"])
            for field_name in params_class.model_fields.keys():
                if field_name not in [
                    "num_classes",
                    "weight_init_fn",
                    "batch_norm",
                    "randomize",
                ]:
                    all_param_names.add(field_name)

        # Define limites unificados
        for param_name in all_param_names:
            if param_name == "dropout_rate":
                bounds[param_name] = (0.0, 0.5)
            elif param_name == "min_channels":
                bounds[param_name] = (16, 64,)
            elif param_name == "max_channels":
                bounds[param_name] = (64, 512,)  
            elif param_name == "num_layers":
                bounds[param_name] = (2, 25)  
            elif param_name == "width_multiplier":
                bounds[param_name] = (0.5, 1.5)
            elif param_name == "resolution_multiplier":
                bounds[param_name] = (0.5, 1.0)
            else:
                bounds[param_name] = (0.0, 1.0)

        bounds["learning_rate"] = (1e-4, 1e-2)

        return bounds

    def _get_architecture_from_vector(
        self, x: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extrai a arquitetura escolhida e seus parâmetros do vetor de otimização.

        Args:
            x (np.ndarray): Vetor de parâmetros completo [architecture_index, param1, param2, ...]

        Returns:
            Tuple[str, Dict[str, Any]]: (nome_da_arquitetura, parâmetros_da_arquitetura)
        """
        # A primeira dimensão é o índice da arquitetura (normalizado entre 0 e 1)
        architecture_index_normalized = x[0]

        # Converte para índice discreto
        n_architectures = len(self.architectures_to_optimize)
        architecture_index = int(architecture_index_normalized * n_architectures)

        # Garante que o índice está dentro dos limites
        architecture_index = max(0, min(architecture_index, n_architectures - 1))

        # Obtém o nome da arquitetura
        architecture_name = self.architectures_to_optimize[architecture_index]

        # Extrai os parâmetros (resto do vetor, exceto o último que é LR)
        params_vector = x[1:-1] 

        architecture_params = self._convert_params_for_architecture(
            params_vector, architecture_name
        )

        return architecture_name, architecture_params

    def _extract_learning_rate(self, x: np.ndarray) -> float:
        """
        Extrai o learning rate do vetor de otimização usando conversão logarítmica.
        
        Args:
            x (np.ndarray): Vetor completo de otimização [architecture_index, param1, ..., learning_rate]
            
        Returns:
            float: Learning rate extraído (em escala logarítmica)
        """
        # O último elemento do vetor é o learning rate normalizado [0, 1]
        lr_normalized = x[-1]
        lr_min, lr_max = self.param_bounds["learning_rate"]
        # log10(lr) = normalized * (log10(max) - log10(min)) + log10(min)
        log_lr = lr_normalized * (np.log10(lr_max) - np.log10(lr_min)) + np.log10(lr_min)
        learning_rate = 10 ** log_lr
        # Garante que está dentro dos limites (por segurança)
        learning_rate = max(lr_min, min(lr_max, learning_rate))
        
        return float(learning_rate)

    def _convert_params_for_architecture(
        self, params_vector: np.ndarray, architecture_name: str
    ) -> Dict[str, Any]:
        """
        Converte um vetor de parâmetros para os parâmetros específicos de uma arquitetura.

        Args:
            params_vector (np.ndarray): Vetor de parâmetros normalizados [0,1]
            architecture_name (str): Nome da arquitetura

        Returns:
            Dict[str, Any]: Parâmetros da arquitetura
        """
        architecture_info = self.all_architectures[architecture_name]
        params_class = type(architecture_info["params"])

        params = {}

        # Converte cada parâmetro do vetor unificado para os parâmetros específicos da arquitetura
        param_index = 0
        for param_name, (min_val, max_val) in self.param_bounds.items():
            if param_name == "learning_rate":
                continue
            if param_name in params_class.model_fields:
                normalized_value = (
                    params_vector[param_index] * (max_val - min_val) + min_val
                )

                # Converte para o tipo correto baseado no campo da classe
                field = params_class.model_fields[param_name]
                field_type = field.annotation

                if field_type == int:
                    params[param_name] = int(round(normalized_value))
                elif field_type == float:
                    params[param_name] = float(normalized_value)
                elif field_type == bool:
                    params[param_name] = bool(round(normalized_value))
                else:
                    params[param_name] = normalized_value

                param_index += 1

        params["num_classes"] = len(self.classes)
        if "batch_norm" in params_class.model_fields:
            params["batch_norm"] = True

        return params

    def _convert_to_architecture_params(
        self, x: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Converte um vetor de parâmetros em arquitetura e seus parâmetros.

        Returns:
            Tuple[str, Dict[str, Any]]: (nome_da_arquitetura, parâmetros)
        """
        return self._get_architecture_from_vector(x)

    def _generate_initial_candidates(
        self,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, Dict[str, float]]]]:
        """
        Fase 1a: Gera candidatos iniciais usando AFSA para diversificar 
        os movimentos do enxame de partículas.
        
        O AFSA explora o espaço de busca de forma inteligente usando comportamentos
        de peixes artificiais (cluster, forrageamento, movimento aleatório) para
        criar uma população inicial diversificada que será usada pelo PSO.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.ndarray, Dict[str, float]]]]: (candidatos, lista de tuplas (candidato, métricas))
        """
        # Inicializa o AFSA 
        afsa = AFSA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            visual=0.5,  # Campo de visão para explorar o espaço
            step=0.1,  # Tamanho do passo para movimentação
            try_times=5,  # Número de tentativas para forrageamento
            max_iter=50,  # Número de iterações para otimização
            lower_bound=0.0,
            upper_bound=1.0,
        )

        # Função de fitness para o AFSA 
        def afsa_fitness(x):
            architecture_name, params = self._convert_to_architecture_params(x)
            diversity_score = 0
            arch_diversity = (
                abs(x[0] - 0.5) * 2
            ) 
            diversity_score += arch_diversity

            # Pula o índice da arquitetura (x[0]) e o learning_rate (x[-1])
            param_vector = x[1:-1]  
            param_index = 0
            for param_name, (min_val, max_val) in self.param_bounds.items():
                if param_name == "learning_rate":
                    # Considera LR na diversidade
                    normalized_value = x[-1]
                elif param_index < len(param_vector):
                    normalized_value = param_vector[param_index]
                    param_index += 1
                else:
                    continue

                # Incentiva exploração de todo o espaço de busca
                edge_bonus = min(normalized_value, 1 - normalized_value) * 2
                diversity_score += (
                    1 - edge_bonus
                )  
                if param_name in ["min_channels", "max_channels", "num_layers"]:
                    # Para parâmetros estruturais, incentiva mais variação
                    diversity_score += abs(normalized_value - 0.5) * 2
                elif param_name == "dropout_rate":
                    # Para dropout, incentiva valores baixos a médios
                    diversity_score += (1 - normalized_value) * 0.5
                elif param_name == "learning_rate":
                    # Para LR, incentiva exploração de diferentes valores
                    diversity_score += abs(normalized_value - 0.5) * 1.5

            # Penaliza soluções muito similares na população atual
            similarity_penalty = 0
            if hasattr(afsa, "population") and len(afsa.population) > 1:
                for other_x in afsa.population:
                    if not np.array_equal(x, other_x):
                        distance = np.linalg.norm(x - other_x)
                        if distance < 0.3:  # Se muito próximos
                            similarity_penalty += (0.3 - distance) * 5

            # Score final: maximiza diversidade e minimiza similaridade
            final_score = diversity_score - similarity_penalty

            return final_score

        self._print_step("Configurando função de fitness do AFSA (baseada em diversidade)")
        afsa.fitness_function = afsa_fitness

        # Executa o AFSA para gerar candidatos
        self._print_step(f"Executando AFSA por {self.afsa_params['max_iter']} iterações")
        candidates = afsa.optimize()

        print(f"\n✅ AFSA concluído!")
        print(f"   • {len(candidates)} candidatos gerados com diversidade de arquiteturas")
        architectures_used = set()
        for candidate in candidates:
            architecture_name, _ = self._convert_to_architecture_params(candidate)
            architectures_used.add(architecture_name)
        print(f"  • Arquiteturas exploradas: {list(architectures_used)}")
        print(f"  • Parâmetros otimizados: {len(self.param_bounds)} parâmetros")

        self._print_section("WARM-UP: Treinando e avaliando candidatos AFSA")
        candidates_metrics = []
        for i, candidate in enumerate(tqdm(candidates, desc="Warm-up"), 1):
            print(f"\n🔄 Avaliando candidato {i}/{len(candidates)}")
            metrics = self._warm_up_candidate(candidate)
            candidates_metrics.append((candidate, metrics))
            arch_name, arch_params = self._convert_to_architecture_params(candidate)
            oace_score = self._calculate_oace_score(metrics)
            self._print_candidate_details(candidate, metrics, arch_name, arch_params, oace_score, i)

        return candidates, candidates_metrics

    def _warm_up_candidate(self, candidate_vector: np.ndarray) -> Dict[str, float]:
        """
        Realiza o warm-up de um candidato e retorna suas métricas.
        Implementa cache para evitar re-avaliação de candidatos idênticos.

        Args:
            candidate_vector (np.ndarray): Vetor completo do candidato.

        Returns:
            Dict[str, float]: Métricas do candidato após o warm-up.
        """

        candidate_key = tuple(np.round(candidate_vector, decimals=4))
        if candidate_key in self.candidates_cache:
            self.cache_hits += 1
            print(f"🎯 Cache HIT! Candidato já avaliado (total hits: {self.cache_hits})")
            return self.candidates_cache[candidate_key]
        
        self.cache_misses += 1
        
        # Extrai learning rate do vetor 
        learning_rate = self._extract_learning_rate(candidate_vector)
        # Extrai arquitetura e parâmetros do vetor (sem o LR)
        architecture_name, architecture_params = self._convert_to_architecture_params(
            candidate_vector
        )

        print(f"   🏗️  Arquitetura: {architecture_name}")
        print(f"   ⚙️  Parâmetros: {architecture_params}")
        print(f"   📈 Learning Rate: {learning_rate:.6f}")

        architecture_info = self.all_architectures[architecture_name]
        params_class = type(architecture_info["params"])
        params = params_class(**architecture_params)

        # Realiza o warm-up
        print(f"   🔥 Iniciando treinamento...")
        test_metrics = architecture_info["warm_up"](
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            classes=self.classes,
            num_epochs=8,
            device=self.device, 
            params=params,
            learning_rate=learning_rate, 
        )

        self.candidates_cache[candidate_key] = test_metrics
        
        print(f"   📊 Métricas obtidas: {test_metrics}")
        print(f"   💾 Resultado salvo no cache. Total cached: {len(self.candidates_cache)}")

        return test_metrics

    def _calculate_metrics_ranges(
        self, candidates_metrics: List[Tuple[np.ndarray, Dict[str, float]]]
    ) -> None:
        """
        Calcula os limites (min/max) para cada métrica baseado nos candidatos e suas métricas já calculadas.

        Args:
            candidates_metrics (List[Tuple[np.ndarray, Dict[str, float]]]): Lista de tuplas (candidato, métricas).
        """
        print("Calculando limites das métricas...")
        all_metrics = [metrics for _, metrics in candidates_metrics]
        self.metrics_history.extend(candidates_metrics)
        # Calcula os limites para cada métrica (excluindo loss por comportamento inverso)
        assertiveness_ranges = {
            "top1_acc": {
                "min": min(m["top1_acc"] for m in all_metrics),
                "max": max(m["top1_acc"] for m in all_metrics),
            },
            "top5_acc": {
                "min": min(m["top5_acc"] for m in all_metrics),
                "max": max(m["top5_acc"] for m in all_metrics),
            },
            "precision_macro": {
                "min": min(m["precision_macro"] for m in all_metrics),
                "max": max(m["precision_macro"] for m in all_metrics),
            },
            "recall_macro": {
                "min": min(m["recall_macro"] for m in all_metrics),
                "max": max(m["recall_macro"] for m in all_metrics),
            },
            "f1_macro": {
                "min": min(m["f1_macro"] for m in all_metrics),
                "max": max(m["f1_macro"] for m in all_metrics),
            },
        }

        cost_ranges = {
            "total_params": {
                "min": min(m["total_params"] for m in all_metrics),
                "max": max(m["total_params"] for m in all_metrics),
            },
            "avg_inference_time": {
                "min": min(m["avg_inference_time"] for m in all_metrics),
                "max": max(m["avg_inference_time"] for m in all_metrics),
            },
            "memory_used_mb": {
                "min": min(m["memory_used_mb"] for m in all_metrics),
                "max": max(m["memory_used_mb"] for m in all_metrics),
            },
            "gflops": {
                "min": min(m["gflops"] for m in all_metrics),
                "max": max(m["gflops"] for m in all_metrics),
            },
        }

        self.metrics_ranges = {
            "assertiveness": assertiveness_ranges,
            "cost": cost_ranges,
        }

    def fitness_function(self, x: np.ndarray) -> float:
        """
        Função de fitness que avalia um candidato usando o score OACE.
        
        Args:
            x (np.ndarray): Vetor de parâmetros normalizados
            
        Returns:
            float: Score OACE (entre 0 e 1)
        """
        cache_key = str(x.tolist())
        if cache_key in self.candidates_cache:
            self.cache_hits += 1
            print(f"🎯 Cache HIT! Candidato já avaliado (total hits: {self.cache_hits})")
            return self.candidates_cache[cache_key]
        self.cache_misses += 1
        print(f"🆕 Novo candidato avaliado")
        
        # Treina e avalia o candidato
        metrics = self._warm_up_candidate(x)
        # Calcula o score OACE
        score = self._calculate_oace_score(metrics)
        # Armazena no cache
        self.candidates_cache[cache_key] = score
        
        print(f"   🎯 Score OACE calculado: {score:.6f}")

        if score > 1.0:
            print(f"   ⚠️  AVISO: Score OACE inválido ({score:.6f}) > 1.0. Corrigindo...")
            score = 1.0
        elif score < 0.0:
            print(f"   ⚠️  AVISO: Score OACE inválido ({score:.6f}) < 0.0. Corrigindo...")
            score = 0.0
        
        return score

    def initialize_components(self):
        """
        Inicializa os componentes do algoritmo híbrido.
        """
        # Inicializa o AFSA
        self.afsa = AFSA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            visual=self.afsa_params["visual"],
            step=self.afsa_params["step"],
            try_times=self.afsa_params["try_times"],
            max_iter=self.afsa_params["max_iter"],
            lower_bound=0.0,  
            upper_bound=1.0,
        )
        # Inicializa o PSO 
        self.pso = PSO(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=self.max_iter,
            lower_bound=0.0,
            upper_bound=1.0,
            afsa_params=self.afsa_params,
            pso_options=self.pso_params,
            logger=self.logger
        )
        
        # Inicializa o GA
        self.ga = GA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=self.max_iter,
            lower_bound=0.0,
            upper_bound=1.0,
            initial_crossover_rate=self.ga_params["initial_crossover_rate"],
            initial_mutation_rate=self.ga_params["initial_mutation_rate"],
            tournament_size=self.ga_params["tournament_size"],
        )

    def optimize(self):
        """
        Executa o processo de otimização híbrida AFSA-GA-PSO seguindo o fluxo correto:
        
        Fase 1: AFSA-PSO (Otimização Inicial)
        - AFSA diversifica movimentos do enxame de partículas
        - PSO com AFSA gera "soluções de otimização inicial"
        
        Fase 2: GA-PSO (Otimização Global)  
        - Melhores soluções da Fase 1 são usadas como população inicial
        - GA-PSO com operadores genéticos refina as soluções
        - Resultado final é a "solução de otimização global"
        
        Returns:
            tuple: (melhor arquitetura encontrada, melhor valor de fitness)
        """
        self._print_header("INICIANDO OTIMIZAÇÃO HÍBRIDA AFSA-GA-PSO")
        self._print_configuration()
        
        try:
            # Passo 1: Inicializa os componentes
            self._print_step("Inicializando componentes do algoritmo híbrido")
            self.initialize_components()

            config = {
                "population_size": self.population_size,
                "max_iter": self.max_iter,
                "lambda_param": self.lambda_param,
                "afsa_params": self.afsa_params,
                "pso_params": self.pso_params,
                "ga_params": self.ga_params,
                "architectures_to_optimize": self.architectures_to_optimize
            }
            self.logger.start_experiment(config)

            # FASE 1: OTIMIZAÇÃO INICIAL COM AFSA-PSO
            self._print_section("FASE 1: OTIMIZAÇÃO INICIAL COM AFSA-PSO")
            
            # Gera população inicial usando AFSA para diversificar movimentos
            self._print_step("Gerando população inicial diversificada com AFSA", 
                           f"Tamanho: {self.population_size}, Iterações: {self.afsa_params['max_iter']}")
            initial_population, candidates_metrics = self._generate_initial_candidates()
            
            # Calcula limites das méctricas usando as métricas já calculadas
            self._print_step("Calculando limites das métricas para normalização OACE")
            self._calculate_metrics_ranges(candidates_metrics)
            
            # Executa PSO com população inicial do AFSA
            self._print_step("Executando PSO com população inicial diversificada", 
                           f"Iterações: {self.max_iter}, Parâmetros: {self.pso_params}")
            phase1_solutions = self._execute_afsa_pso_phase(initial_population, candidates_metrics)
            
            best_idx = np.argmax([self.fitness_function(x) for x in phase1_solutions])
            best_arch, best_params = self._convert_to_architecture_params(phase1_solutions[best_idx])
            best_fitness = self.fitness_function(phase1_solutions[best_idx])
            self._print_phase_summary("AFSA-PSO", best_fitness, best_arch, best_params)

            # FASE 2: OTIMIZAÇÃO GLOBAL COM GA-PSO
            self._print_section("FASE 2: OTIMIZAÇÃO GLOBAL COM GA-PSO")
            
            # Usa as melhores soluções da Fase 1 como população inicial para GA-PSO
            self._print_step("Refinando soluções com GA-PSO usando operadores genéticos", 
                           f"Taxa crossover: {self.ga_params['initial_crossover_rate']}, "
                           f"Taxa mutação: {self.ga_params['initial_mutation_rate']}")
            best_position, best_fitness = self._execute_ga_pso_phase(phase1_solutions)

            # Registra a solução global ótima
            self.best_solution = best_position
            self.best_fitness = best_fitness

            # Converte a melhor solução em arquitetura e parâmetros
            best_architecture_name, best_architecture_params = (
                self._convert_to_architecture_params(best_position)
            )

            # Obtém as métricas finais
            self._print_step("Avaliando solução final para métricas completas")
            final_metrics = self._warm_up_candidate(best_position)
            self.logger.log_final_results(
                best_architecture=best_architecture_name,
                best_params=best_architecture_params,
                best_fitness=best_fitness,
                final_metrics=final_metrics
            )
            self._print_final_results(best_architecture_name, best_architecture_params, 
                                    best_fitness, final_metrics)

            return best_architecture_name, best_architecture_params, best_fitness
            
        except Exception as e:
            print(f"\n❌ Erro durante a otimização: {str(e)}")
            if hasattr(self, 'logger'):
                try:
                    self.logger._save_log()
                    print(f"\n✓ Logs parciais salvos em: {self.logger.log_dir}/{self.logger.current_experiment}")
                except:
                    pass
            raise

    def _execute_afsa_pso_phase(self, initial_population, candidates_metrics):
        """
        Executa a Fase 1: AFSA-PSO (Otimização Inicial) - VERSÃO INTEGRADA
        
        Conforme o artigo, o AFSA otimiza o PSO aplicando comportamentos
        (cluster, foraging, random) nas partículas do enxame. Após cada modificação
        do AFSA, o PSO executa iterações para refinar as soluções.
        
        Fluxo:
        1. Calcula fitness inicial dos candidatos (warm-up)
        2. Inicializa PSO com população inicial
        3. Configura AFSA para usar função de fitness do PSO (OACE)
        4. Loop alternado: AFSA modifica partículas → PSO executa iterações
        5. Retorna melhores soluções encontradas
        """
        self._print_section("AFSA-PSO: Iniciando Fase 1 com otimização integrada")
        
        # 1. Calcula fitness inicial dos candidatos
        self._print_step("Calculando fitness dos candidatos iniciais com OACE")
        initial_fitness = []
        for i, (candidate, metrics) in enumerate(candidates_metrics, 1):
            print(f"\n🔄 Avaliando candidato {i}/{len(candidates_metrics)}")
            score = self._calculate_oace_score(metrics)
            initial_fitness.append(score)
            print(f"   🎯 Score OACE: {score:.6f}")

        initial_fitness = np.array(initial_fitness)
        best_idx = np.argmax(initial_fitness)
        
        self._print_population_summary(initial_population, initial_fitness, "AFSA-PSO Inicial")
        
        print(f"\n🏆 Melhor candidato inicial:")
        print(f"   • Índice: {best_idx}")
        print(f"   • Score OACE: {initial_fitness[best_idx]:.6f}")
        arch_name, arch_params = self._convert_to_architecture_params(initial_population[best_idx])
        print(f"   • Arquitetura: {arch_name}")
        print(f"   • Parâmetros: {arch_params}")
        self.logger.log_iteration(
            iteration=0,
            phase="AFSA-PSO",
            population=initial_population,
            fitness_values=initial_fitness,
            best_position=initial_population[best_idx],
            best_fitness=initial_fitness[best_idx],
            metrics=candidates_metrics[best_idx][1],
            oace_score=initial_fitness[best_idx]
        )

        # 2. Configura função de fitness para o PSO
        def pso_fitness_function(x):
            """Função de fitness para o PSO (minimização)"""
            if x.ndim == 1:
                return -self.fitness_function(x)
            else:
                scores = []
                for xi in x:
                    score = self.fitness_function(xi)
                    scores.append(score)
                return -np.array(scores)

        self.pso.fitness_function = pso_fitness_function
        
        # 3. Inicializa PSO com população inicial
        self._print_step("Inicializando enxame PSO com população inicial")
        self.pso.initialize_swarm_with_population(initial_population)
        
        # 4. Configura AFSA para trabalhar sobre PSO
        # Reutiliza instância do AFSA existente ou cria nova se necessário
        if self.afsa is None:
            self.afsa = AFSA(
                population_size=self.population_size,
                n_dim=self.n_dim,
                visual=self.afsa_params.get('visual', 0.5),
                step=self.afsa_params.get('step', 0.1),
                try_times=self.afsa_params.get('try_times', 5),
                max_iter=self.afsa_params.get('max_iter', 3), 
                lower_bound=self.pso.lower_bound,
                upper_bound=self.pso.upper_bound
            )
        else:
            self.afsa.population_size = self.population_size
            self.afsa.n_dim = self.n_dim
            self.afsa.visual = self.afsa_params.get('visual', 0.5)
            self.afsa.step = self.afsa_params.get('step', 0.1)
            self.afsa.try_times = self.afsa_params.get('try_times', 5)
            self.afsa.lower_bound = self.pso.lower_bound
            self.afsa.upper_bound = self.pso.upper_bound
        
        # Configura AFSA para usar função de fitness do PSO 
        self.afsa.fitness_function = self.fitness_function
        
        # 5. Loop alternado: AFSA modifica → PSO executa
        self._print_step("Iniciando loop alternado AFSA-PSO", 
                        f"Iterações AFSA: {self.afsa_params.get('max_iter', 3)}, "
                        f"Iterações PSO por ciclo: 1")
        
        max_afsa_iter = self.afsa_params.get('max_iter', 3)
        
        for afsa_iter in range(max_afsa_iter):
            print(f"\n🔄 AFSA-PSO Iteração {afsa_iter + 1}/{max_afsa_iter}")
            
            # 5a. AFSA aplica comportamentos nas partículas do PSO
            self._print_step(f"AFSA aplicando comportamentos (iteração {afsa_iter + 1})")
            pso_fitness_before_afsa = -self.pso.optimizer.swarm.pbest_cost.copy()  # Converte para OACE
            pso_gbest_before_afsa = -float(self.pso.optimizer.swarm.best_cost)
            
            print(f"\n      🔄 PSO ANTES DO AFSA:")
            print(f"         • Melhor OACE (gbest): {pso_gbest_before_afsa:.6f}")
            print(f"         • OACE médio (pbest): {np.mean(pso_fitness_before_afsa):.6f}")

            afsa_optimized_population = self._apply_afsa_behaviors_to_pso(afsa_iter)
            
            # 5b. Atualiza enxame do PSO com população otimizada pelo AFSA
            print(f"\n      🔄 ATUALIZANDO ENXAME DO PSO COM POPULAÇÃO DO AFSA")
            print(f"         • Substituindo {len(afsa_optimized_population)} partículas do PSO")
            print(f"         • Partículas antigas do PSO foram substituídas pelas do AFSA")
            print(f"         • Recalculando fitness das partículas modificadas...")
            
            self.pso.optimizer.swarm.position = afsa_optimized_population.copy()
    
            # 5c. Recalcula fitness das partículas modificadas
            fitness_values = self.pso.fitness_function(afsa_optimized_population)
            
            # 5d. Atualiza pbest se necessário (PSO usa minimização)
            pbest_updated = 0
            for i in range(len(afsa_optimized_population)):
                # PSO minimiza, então compara custos negativos
                if fitness_values[i] < self.pso.optimizer.swarm.pbest_cost[i]:
                    self.pso.optimizer.swarm.pbest_pos[i] = afsa_optimized_population[i].copy()
                    self.pso.optimizer.swarm.pbest_cost[i] = fitness_values[i]
                    pbest_updated += 1
            
            # 5e. Atualiza gbest se necessário
            best_idx = np.argmin(self.pso.optimizer.swarm.pbest_cost)
            new_gbest_cost = self.pso.optimizer.swarm.pbest_cost[best_idx]
            
            if new_gbest_cost < self.pso.optimizer.swarm.best_cost:
                old_gbest = -float(self.pso.optimizer.swarm.best_cost)
                self.pso.optimizer.swarm.best_pos = self.pso.optimizer.swarm.pbest_pos[best_idx].copy()
                self.pso.optimizer.swarm.best_cost = new_gbest_cost
                new_gbest = -float(self.pso.optimizer.swarm.best_cost)
                print(f"         • pbest atualizado: {pbest_updated}/{len(afsa_optimized_population)} partículas")
                print(f"         • ✅ gbest ATUALIZADO: {old_gbest:.6f} → {new_gbest:.6f} ({'+' if new_gbest > old_gbest else ''}{new_gbest - old_gbest:.6f})")
            else:
                print(f"         • pbest atualizado: {pbest_updated}/{len(afsa_optimized_population)} partículas")
                print(f"         • gbest mantido: {-float(self.pso.optimizer.swarm.best_cost):.6f}")
            
            # 5f. PSO executa 1 iteração para refinar
            print(f"\n      ⚙️  PSO REFINANDO SOLUÇÕES MODIFICADAS PELO AFSA")
            print(f"         • Executando 1 iteração do PSO...")
            
            pso_gbest_before_refine = -float(self.pso.optimizer.swarm.best_cost)
            pso_fitness_before_refine = -np.mean(self.pso.optimizer.swarm.pbest_cost)
            
            try:
                self.pso._update_swarm_one_iteration()
                print(f"         • ✅ PSO executou 1 iteração com sucesso")
            except Exception as e:
                print(f"         • ⚠️  Erro ao executar PSO: {e}")
            
            pso_gbest_after_refine = -float(self.pso.optimizer.swarm.best_cost)
            pso_fitness_after_refine = -np.mean(self.pso.optimizer.swarm.pbest_cost)
            
            print(f"\n      📊 RESULTADO DO REFINAMENTO DO PSO:")
            print(f"         • gbest ANTES do refinamento: {pso_gbest_before_refine:.6f}")
            print(f"         • gbest DEPOIS do refinamento: {pso_gbest_after_refine:.6f}")
            print(f"         • Mudança no gbest: {pso_gbest_after_refine - pso_gbest_before_refine:+.6f}")
            print(f"         • Fitness médio ANTES: {pso_fitness_before_refine:.6f}")
            print(f"         • Fitness médio DEPOIS: {pso_fitness_after_refine:.6f}")
            print(f"         • Mudança no fitness médio: {pso_fitness_after_refine - pso_fitness_before_refine:+.6f}")
            
            print(f"\n      📈 RESUMO DO CICLO AFSA-PSO (Iteração {afsa_iter + 1}):")
            print(f"         • OACE inicial (antes do AFSA): {pso_gbest_before_afsa:.6f}")
            print(f"         • OACE após AFSA: {pso_gbest_after_refine:.6f}")
            print(f"         • OACE após PSO refinar: {pso_gbest_after_refine:.6f}")
            print(f"         • Melhoria total no ciclo: {pso_gbest_after_refine - pso_gbest_before_afsa:+.6f}")
            print(f"      {'='*60}")
            
            # Log da iteração
            current_population = self.pso.optimizer.swarm.position
            current_fitness = np.array([self.fitness_function(p) for p in current_population])
            self.logger.log_iteration(
                iteration=afsa_iter + 1,
                phase="AFSA-PSO",
                population=current_population,
                fitness_values=current_fitness,
                best_position=self.pso.optimizer.swarm.best_pos,
                best_fitness=-float(self.pso.optimizer.swarm.best_cost),
                metrics=None, 
                oace_score=-float(self.pso.optimizer.swarm.best_cost)
            )
        
        # 6. Retorna melhores soluções do PSO
        print(f"\n✅ AFSA-PSO concluído!")
        best_pos = self.pso.optimizer.swarm.best_pos
        best_oace = -float(self.pso.optimizer.swarm.best_cost)
        print(f"   • Melhor posição encontrada: {best_pos}")
        print(f"   • Melhor score OACE: {best_oace:.6f}")
        
        # Seleciona as melhores partículas (pbest)
        phase1_solutions = self.pso.optimizer.swarm.pbest_pos.copy()
        final_fitness = np.array([self.fitness_function(p) for p in phase1_solutions])
        
        # Mostra resumo final da Fase 1
        self._print_population_summary(phase1_solutions, final_fitness, "AFSA-PSO Final")
        
        print(f"\n✅ Fase AFSA-PSO Concluída!")
        print(f"   • Melhor score da Fase 1: {np.max(final_fitness):.6f}")
        print(f"   • {len(phase1_solutions)} soluções selecionadas para Fase 2")
        
        return phase1_solutions

    def _apply_ga_operators_to_pso(self, iteration):
        """
        Aplica operadores genéticos do GA nas partículas do PSO.
        
        Conforme o artigo, o GA otimiza o PSO aplicando crossover e mutação
        nas partículas do enxame, gerando novas soluções que serão refinadas pelo PSO.
        
        Args:
            iteration: Número da iteração atual (para taxas adaptativas)
        
        Returns:
            np.ndarray: População otimizada pelo GA (formato: (population_size, n_dim))
        """
        # 1. Obtém partículas atuais do PSO
        pso_instance = getattr(self, 'pso_phase2', self.pso)
        current_particles = pso_instance.optimizer.swarm.position.copy()
        
        # LOG: Estado ANTES do GA
        print(f"\n      {'='*60}")
        print(f"      📋 ESTADO ANTES DO GA APLICAR OPERADORES")
        print(f"      {'='*60}")
        print(f"      • Número de partículas: {len(current_particles)}")
        
        # Calcula fitness ANTES
        fitness_before = np.array([self.fitness_function(p) for p in current_particles])
        print(f"      • Fitness ANTES (OACE): {[f'{f:.6f}' for f in fitness_before]}")
        print(f"      • Melhor fitness ANTES: {np.max(fitness_before):.6f}")
        print(f"      • Fitness médio ANTES: {np.mean(fitness_before):.6f}")
        
        # 2. Calcula taxas adaptativas do GA
        crossover_rate = self.ga.adaptive_crossover_rate(iteration)
        mutation_rate = self.ga.adaptive_mutation_rate(iteration)
        
        print(f"\n      🔬 GA APLICANDO OPERADORES GENÉTICOS")
        print(f"      • Taxa Crossover: {crossover_rate:.3f}")
        print(f"      • Taxa Mutação: {mutation_rate:.3f}")
        print(f"      • Iteração: {iteration + 1}")
        
        # 3. Converte partículas para formato do GA (Individual do DEAP)
        from deap import creator
        population_individuals = []
        for particle in current_particles:
            particle_array = np.array(particle).copy()
            ind = creator.Individual(particle_array)
            try:
                fitness_value = self.fitness_function(particle_array)
                if isinstance(fitness_value, (int, float)):
                    ind.fitness.values = (fitness_value,)
                elif isinstance(fitness_value, tuple):
                    ind.fitness.values = fitness_value
                else:
                    ind.fitness.values = (float(fitness_value),)
            except Exception as e:
                print(f"      ⚠️  Erro ao avaliar fitness: {e}")
                ind.fitness.values = (0.0,)
            population_individuals.append(ind)
        
        # 4. Aplica operadores genéticos usando DEAP varOr
        print(f"      • Aplicando varOr com {len(population_individuals)} indivíduos...")
        
        from deap import algorithms
        offspring = algorithms.varOr(
            population_individuals,
            self.ga.toolbox,
            lambda_=len(population_individuals), 
            cxpb=crossover_rate,
            mutpb=mutation_rate
        )
        
        print(f"      ✅ varOr gerou {len(offspring)} novos indivíduos")

        # Conta quantos foram modificados
        num_crossover = 0
        num_mutation = 0
        num_unchanged = 0
        
        # 5. Avalia fitness dos novos indivíduos gerados
        for idx, ind in enumerate(offspring):
            if not ind.fitness.valid:
                try:
                    fitness_value = self.fitness_function(np.array(ind))
                    if isinstance(fitness_value, (int, float)):
                        ind.fitness.values = (fitness_value,)
                    elif isinstance(fitness_value, tuple):
                        ind.fitness.values = fitness_value
                    else:
                        ind.fitness.values = (float(fitness_value),)
                except Exception as e:
                    print(f"      ⚠️  Erro ao avaliar fitness do offspring {idx}: {e}")
                    ind.fitness.values = (0.0,)
            
            original = current_particles[idx]
            new = np.array(ind)
            if not np.allclose(original, new, atol=1e-6):
                # Verifica se foi crossover ou mutação (heurística simples)
                if np.sum(np.abs(original - new)) > 0.1:
                    num_crossover += 1
                else:
                    num_mutation += 1
            else:
                num_unchanged += 1
        
        print(f"      📊 Modificações detectadas:")
        print(f"         • Crossover aplicado: ~{num_crossover} partículas")
        print(f"         • Mutação aplicada: ~{num_mutation} partículas")
        print(f"         • Sem modificação: ~{num_unchanged} partículas")
        
        # 6. Converte de volta para numpy array
        optimized_particles = np.array([np.array(ind) for ind in offspring])
        
        # 7. Garante que as partículas estão dentro dos limites
        pso_instance = getattr(self, 'pso_phase2', self.pso)
        optimized_particles = np.clip(
            optimized_particles,
            pso_instance.lower_bound,
            pso_instance.upper_bound
        )
        
        # 📊 LOG: Estado DEPOIS do GA
        print(f"\n      📋 ESTADO DEPOIS DO GA APLICAR OPERADORES")
        print(f"      {'='*60}")
        
        # Calcula fitness DEPOIS
        fitness_after = np.array([self.fitness_function(p) for p in optimized_particles])
        print(f"      • Fitness DEPOIS (OACE): {[f'{f:.6f}' for f in fitness_after]}")
        print(f"      • Melhor fitness DEPOIS: {np.max(fitness_after):.6f}")
        print(f"      • Fitness médio DEPOIS: {np.mean(fitness_after):.6f}")
        
        # Comparação
        improvement = np.max(fitness_after) - np.max(fitness_before)
        avg_improvement = np.mean(fitness_after) - np.mean(fitness_before)
        
        print(f"\n      📈 COMPARAÇÃO ANTES vs DEPOIS:")
        print(f"         • Melhor fitness: {np.max(fitness_before):.6f} → {np.max(fitness_after):.6f} "
              f"({'+' if improvement >= 0 else ''}{improvement:.6f})")
        print(f"         • Fitness médio: {np.mean(fitness_before):.6f} → {np.mean(fitness_after):.6f} "
              f"({'+' if avg_improvement >= 0 else ''}{avg_improvement:.6f})")
        
        # Verifica mudanças nas partículas
        changes = []
        for i in range(len(current_particles)):
            diff = np.linalg.norm(current_particles[i] - optimized_particles[i])
            changes.append(diff)
            if diff > 1e-6:
                print(f"         • Partícula {i+1}: modificada (distância: {diff:.6f})")
        
        if all(c < 1e-6 for c in changes):
            print(f"         ⚠️  AVISO: Nenhuma partícula foi modificada significativamente!")
        else:
            print(f"         ✅ {sum(1 for c in changes if c > 1e-6)} partículas modificadas")
        
        print(f"      {'='*60}\n")
        
        return optimized_particles

    def _apply_afsa_behaviors_to_pso(self, iteration):
        """
        Aplica comportamentos do AFSA nas partículas do PSO.
        
        Conforme o artigo, o AFSA otimiza o PSO aplicando comportamentos
        (cluster, foraging, random) nas partículas do enxame, gerando novas
        soluções que serão refinadas pelo PSO.
        
        Args:
            iteration: Número da iteração atual (para logging)
        
        Returns:
            np.ndarray: Partículas otimizadas pelo AFSA (formato: (population_size, n_dim))
        """
        # 1. Obtém partículas atuais do PSO
        current_particles = self.pso.optimizer.swarm.position.copy()
        
        # 📊 LOG: Estado ANTES do AFSA
        print(f"\n      {'='*60}")
        print(f"      📋 ESTADO ANTES DO AFSA APLICAR COMPORTAMENTOS")
        print(f"      {'='*60}")
        print(f"      • Número de partículas: {len(current_particles)}")
        
        # Calcula fitness ANTES
        fitness_before = np.array([self.fitness_function(p) for p in current_particles])
        print(f"      • Fitness ANTES (OACE): {[f'{f:.6f}' for f in fitness_before]}")
        print(f"      • Melhor fitness ANTES: {np.max(fitness_before):.6f}")
        print(f"      • Fitness médio ANTES: {np.mean(fitness_before):.6f}")
        
        print(f"\n      🐟 AFSA APLICANDO COMPORTAMENTOS")
        print(f"      • Iteração: {iteration + 1}")
        print(f"      • Visual: {self.afsa.visual}")
        print(f"      • Step: {self.afsa.step}")
        print(f"      • Try times: {self.afsa.try_times}")
        
        # 2. Para cada partícula, aplica comportamentos do AFSA
        optimized_particles = []
        behaviors_applied = {'cluster': 0, 'foraging': 0, 'random': 0, 'unchanged': 0}
        
        for i, particle in enumerate(current_particles):
            original_particle = particle.copy()
            current_fitness = self.fitness_function(particle)
            
            # Aplica apenas 1 comportamento por partícula 
            best_pos = original_particle
            best_fitness = current_fitness
            best_behavior = 'unchanged'
            
            # 1. Tenta cluster behavior primeiro
            cluster_pos = self.afsa.cluster_behavior_on_particle(
                particle, current_particles, i, self.fitness_function
            )
            cluster_fitness = self.fitness_function(cluster_pos)
            
            if cluster_fitness > best_fitness:
                best_pos = cluster_pos
                best_fitness = cluster_fitness
                best_behavior = 'cluster'
            
            # 2. Se cluster não melhorou o suficiente, tenta foraging
            if best_behavior != 'cluster':
                foraging_pos = self.afsa.foraging_behavior_on_particle(particle, self.fitness_function)
                foraging_fitness = self.fitness_function(foraging_pos)
                
                if foraging_fitness > best_fitness:
                    best_pos = foraging_pos
                    best_fitness = foraging_fitness
                    best_behavior = 'foraging'
            
            # 3. Se nenhum melhorou, aplica random
            if best_behavior == 'unchanged':
                random_pos = self.afsa.random_behavior_on_particle(particle)
                random_fitness = self.fitness_function(random_pos)
                
                if random_fitness > best_fitness:
                    best_pos = random_pos
                    best_fitness = random_fitness
                    best_behavior = 'random'
                else:
                    # Mantém original se nenhum comportamento melhorou
                    best_pos = original_particle
                    best_behavior = 'unchanged'
            
            # Registra comportamento aplicado
            behaviors_applied[best_behavior] += 1
            optimized_particles.append(best_pos)
        
        optimized_particles = np.array(optimized_particles)
        
        # 3. Garante que as partículas estão dentro dos limites
        optimized_particles = np.clip(
            optimized_particles,
            self.pso.lower_bound,
            self.pso.upper_bound
        )
        
        print(f"      📊 Comportamentos aplicados:")
        print(f"         • Cluster: {behaviors_applied['cluster']} partículas")
        print(f"         • Foraging: {behaviors_applied['foraging']} partículas")
        print(f"         • Random: {behaviors_applied['random']} partículas")
        print(f"         • Sem modificação: {behaviors_applied['unchanged']} partículas")
        
        # LOG: Estado DEPOIS do AFSA
        print(f"\n      📋 ESTADO DEPOIS DO AFSA APLICAR COMPORTAMENTOS")
        print(f"      {'='*60}")
        
        # Calcula fitness DEPOIS
        fitness_after = np.array([self.fitness_function(p) for p in optimized_particles])
        print(f"      • Fitness DEPOIS (OACE): {[f'{f:.6f}' for f in fitness_after]}")
        print(f"      • Melhor fitness DEPOIS: {np.max(fitness_after):.6f}")
        print(f"      • Fitness médio DEPOIS: {np.mean(fitness_after):.6f}")
        
        # Comparação
        improvement = np.max(fitness_after) - np.max(fitness_before)
        avg_improvement = np.mean(fitness_after) - np.mean(fitness_before)
        
        print(f"\n      📈 COMPARAÇÃO ANTES vs DEPOIS:")
        print(f"         • Melhor fitness: {np.max(fitness_before):.6f} → {np.max(fitness_after):.6f} "
              f"({'+' if improvement >= 0 else ''}{improvement:.6f})")
        print(f"         • Fitness médio: {np.mean(fitness_before):.6f} → {np.mean(fitness_after):.6f} "
              f"({'+' if avg_improvement >= 0 else ''}{avg_improvement:.6f})")
        
        # Verifica mudanças nas partículas
        changes = []
        for i in range(len(current_particles)):
            diff = np.linalg.norm(current_particles[i] - optimized_particles[i])
            changes.append(diff)
            if diff > 1e-6:
                print(f"         • Partícula {i+1}: modificada (distância: {diff:.6f})")
        
        if all(c < 1e-6 for c in changes):
            print(f"         ⚠️  AVISO: Nenhuma partícula foi modificada significativamente!")
        else:
            print(f"         ✅ {sum(1 for c in changes if c > 1e-6)} partículas modificadas")
        
        print(f"      {'='*60}\n")
        
        return optimized_particles

    def _execute_ga_pso_phase(self, phase1_solutions):
        """
        Executa a Fase 2: GA-PSO (Otimização Global)
        
        Conforme o artigo, o GA otimiza o PSO aplicando operadores genéticos
        (crossover e mutação) nas partículas do enxame. Após cada modificação
        do GA, o PSO executa algumas iterações para refinar as soluções.
        
        Fluxo:
        1. Inicializa PSO com soluções da Fase 1
        2. Loop alternado: GA modifica partículas → PSO executa iterações
        3. Retorna melhor solução encontrada pelo PSO
        
        Args:
            phase1_solutions: Soluções de otimização inicial da Fase 1
            
        Returns:
            tuple: (melhor posição, melhor fitness)
        """
        self._print_section("GA-PSO: Iniciando Fase 2 com soluções da Fase 1")
        
        print(f"   • Shape das soluções: {np.array(phase1_solutions).shape}")
        print(f"   • Primeira solução: {phase1_solutions[0]}")
        
        # Configura função de fitness para o PSO 
        def pso_fitness_function(x):
            """Função de fitness para o PSO (minimização)"""
            if x.ndim == 1:
                return -self.fitness_function(x)
            else:
                scores = []
                for xi in x:
                    score = self.fitness_function(xi)
                    scores.append(score)
                return -np.array(scores)
        
        # 1. Inicializa PSO com soluções da Fase 1
        self._print_step("Inicializando PSO para Fase 2 com soluções da Fase 1")
        # Cria novo PSO para Fase 2 (pode reutilizar parâmetros, mas é uma instância separada)
        pso_phase2 = PSO(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=1, 
            lower_bound=0.0,
            upper_bound=1.0,
            afsa_params=None,  # Não usa AFSA na Fase 2
            pso_options=self.pso_params,
            logger=self.logger
        )
        pso_phase2.fitness_function = pso_fitness_function
        pso_phase2.initialize_swarm_with_population(phase1_solutions)
        
        # Usa pso_phase2 para a Fase 2
        self.pso_phase2 = pso_phase2
        
        # Avalia fitness inicial
        initial_fitness = np.array([self.fitness_function(x) for x in phase1_solutions])
        initial_pso_fitness = -initial_fitness  
        
        best_idx = np.argmax(initial_fitness)
        best_metrics = self._warm_up_candidate(phase1_solutions[best_idx])
        self._print_population_summary(phase1_solutions, initial_fitness, "GA-PSO Inicial")
        
        print(f"\n🏆 Melhor solução inicial da Fase 1:")
        print(f"   • Índice: {best_idx}")
        print(f"   • Score OACE: {initial_fitness[best_idx]:.6f}")
        
        best_architecture, _ = self._convert_to_architecture_params(phase1_solutions[best_idx])
        print(f"   • Arquitetura: {best_architecture}")
        
        # Registra a iteração inicial
        self.logger.log_iteration(
            iteration=0,
            phase="GA-PSO",
            population=phase1_solutions,
            fitness_values=initial_fitness,
            best_position=phase1_solutions[best_idx],
            best_fitness=initial_fitness[best_idx],
            metrics=best_metrics,
            oace_score=initial_fitness[best_idx]
        )
        
        # 2. Loop alternado: GA modifica → PSO executa
        self._print_step("Iniciando loop alternado GA-PSO", 
                        f"Iterações GA: {self.ga_params['max_iter']}, "
                        f"Iterações PSO por ciclo: 1")
        
        max_ga_iter = self.ga_params.get('max_iter', self.max_iter)
        
        for ga_iter in range(max_ga_iter):
            print(f"\n🔄 GA-PSO Iteração {ga_iter + 1}/{max_ga_iter}")
            
            # 2a. GA aplica operadores genéticos nas partículas do PSO
            self._print_step(f"GA aplicando operadores genéticos (iteração {ga_iter + 1})")
            
            # LOG: Estado do PSO ANTES do GA
            pso_instance = getattr(self, 'pso_phase2', self.pso)
            pso_fitness_before_ga = -pso_instance.optimizer.swarm.pbest_cost.copy()  # Converte para OACE
            pso_gbest_before_ga = -float(pso_instance.optimizer.swarm.best_cost)
            
            print(f"\n      🔄 PSO ANTES DO GA:")
            print(f"         • Melhor OACE (gbest): {pso_gbest_before_ga:.6f}")
            print(f"         • OACE médio (pbest): {np.mean(pso_fitness_before_ga):.6f}")
            
            ga_optimized_population = self._apply_ga_operators_to_pso(ga_iter)
            
            # 2b. Atualiza enxame do PSO com população otimizada pelo GA
            print(f"\n      🔄 ATUALIZANDO ENXAME DO PSO COM POPULAÇÃO DO GA")
            print(f"         • Substituindo {len(ga_optimized_population)} partículas do PSO")
            print(f"         • Partículas antigas do PSO foram substituídas pelas do GA")
            
            pso_instance.optimizer.swarm.position = ga_optimized_population.copy()
            
            # 2c. Recalcula fitness das partículas modificadas
            print(f"         • Recalculando fitness das partículas modificadas...")
            fitness_values = pso_instance.fitness_function(ga_optimized_population)
            
            # 2d. Atualiza pbest se necessário (PSO usa minimização)
            pbest_updates = 0
            for i in range(len(ga_optimized_population)):
                if fitness_values[i] < pso_instance.optimizer.swarm.pbest_cost[i]:
                    pso_instance.optimizer.swarm.pbest_pos[i] = ga_optimized_population[i].copy()
                    pso_instance.optimizer.swarm.pbest_cost[i] = fitness_values[i]
                    pbest_updates += 1
            
            print(f"         • pbest atualizado: {pbest_updates}/{len(ga_optimized_population)} partículas")
        
            # 2e. Atualiza gbest
            gbest_updated = False
            if pso_instance.optimizer.swarm.pbest_cost.size > 0:
                best_idx_pso = np.argmin(pso_instance.optimizer.swarm.pbest_cost)
                if pso_instance.optimizer.swarm.pbest_cost[best_idx_pso] < pso_instance.optimizer.swarm.best_cost:
                    old_gbest = -float(pso_instance.optimizer.swarm.best_cost)
                    pso_instance.optimizer.swarm.best_pos = pso_instance.optimizer.swarm.pbest_pos[best_idx_pso].copy()
                    pso_instance.optimizer.swarm.best_cost = pso_instance.optimizer.swarm.pbest_cost[best_idx_pso]
                    new_gbest = -float(pso_instance.optimizer.swarm.best_cost)
                    gbest_updated = True
                    print(f"         • ✅ gbest ATUALIZADO: {old_gbest:.6f} → {new_gbest:.6f} "
                          f"(+{new_gbest - old_gbest:.6f})")
                else:
                    print(f"         • gbest mantido: {(-float(pso_instance.optimizer.swarm.best_cost)):.6f}")
        
            # 2f. PSO executa 1 iteração para refinar as soluções modificadas pelo GA
            print(f"\n      ⚙️  PSO REFINANDO SOLUÇÕES MODIFICADAS PELO GA")
            print(f"         • Executando 1 iteração do PSO...")
            
            # Estado ANTES do PSO refinar
            pso_fitness_before_refine = -pso_instance.optimizer.swarm.pbest_cost.copy()
            pso_gbest_before_refine = -float(pso_instance.optimizer.swarm.best_cost)
            
            self._print_step("PSO refinando soluções modificadas pelo GA")
            try:
                pso_instance._update_swarm_one_iteration()
                print(f"         ✅ PSO executou 1 iteração com sucesso")
            except Exception as e:
                print(f"         ⚠️  Erro na iteração do PSO: {e}")
                # Fallback: apenas atualiza fitness sem mover partículas
                fitness_values = pso_instance.fitness_function(pso_instance.optimizer.swarm.position)
                for i in range(len(pso_instance.optimizer.swarm.position)):
                    if fitness_values[i] < pso_instance.optimizer.swarm.pbest_cost[i]:
                        pso_instance.optimizer.swarm.pbest_pos[i] = pso_instance.optimizer.swarm.position[i].copy()
                        pso_instance.optimizer.swarm.pbest_cost[i] = fitness_values[i]
            
            # Estado DEPOIS do PSO refinar
            pso_fitness_after_refine = -pso_instance.optimizer.swarm.pbest_cost.copy()
            pso_gbest_after_refine = -float(pso_instance.optimizer.swarm.best_cost)
        
            print(f"\n      📊 RESULTADO DO REFINAMENTO DO PSO:")
            print(f"         • gbest ANTES do refinamento: {pso_gbest_before_refine:.6f}")
            print(f"         • gbest DEPOIS do refinamento: {pso_gbest_after_refine:.6f}")
            refine_improvement = pso_gbest_after_refine - pso_gbest_before_refine
            print(f"         • Mudança no gbest: {('+' if refine_improvement >= 0 else '')}{refine_improvement:.6f}")
            
            print(f"         • Fitness médio ANTES: {np.mean(pso_fitness_before_refine):.6f}")
            print(f"         • Fitness médio DEPOIS: {np.mean(pso_fitness_after_refine):.6f}")
            avg_refine_improvement = np.mean(pso_fitness_after_refine) - np.mean(pso_fitness_before_refine)
            print(f"         • Mudança no fitness médio: {('+' if avg_refine_improvement >= 0 else '')}{avg_refine_improvement:.6f}")
            print(f"\n      📈 RESUMO DO CICLO GA-PSO (Iteração {ga_iter + 1}):")
            print(f"         • OACE inicial (antes do GA): {pso_gbest_before_ga:.6f}")
            print(f"         • OACE após GA: {pso_gbest_before_refine:.6f}")
            print(f"         • OACE após PSO refinar: {pso_gbest_after_refine:.6f}")
            total_improvement = pso_gbest_after_refine - pso_gbest_before_ga
            print(f"         • Melhoria total no ciclo: {('+' if total_improvement >= 0 else '')}{total_improvement:.6f}")
            print(f"      {'='*60}\n")
            
            # 2g. Calcula OACE para logging 
            current_oace_fitness = -pso_instance.optimizer.swarm.pbest_cost
            best_oace_score = -float(pso_instance.optimizer.swarm.best_cost)
            
            # 2h. Logging da iteração GA-PSO
            if self.logger:
                # Obtém métricas do melhor candidato atual
                best_pos_current = pso_instance.optimizer.swarm.best_pos
                try:
                    best_metrics_current = self._warm_up_candidate(best_pos_current)
                except:
                    best_metrics_current = None
                
        self.logger.log_iteration(
                    iteration=ga_iter + 1,
                    phase="GA-PSO",
                    population=pso_instance.optimizer.swarm.position,
                    fitness_values=current_oace_fitness,
                    best_position=best_pos_current,
                    best_fitness=best_oace_score,
                    metrics=best_metrics_current,
                    oace_score=best_oace_score,
                    pbest_pos=pso_instance.optimizer.swarm.pbest_pos,
                    pbest_cost=-pso_instance.optimizer.swarm.pbest_cost,  
                    gbest_pos=pso_instance.optimizer.swarm.best_pos,
                    gbest_cost=best_oace_score
                )
            
        print(f"         • Melhor OACE atual: {best_oace_score:.6f}")
        
        # 3. Retorna melhor solução do PSO
        pso_instance = getattr(self, 'pso_phase2', self.pso)
        best_pos = pso_instance.optimizer.swarm.best_pos.copy()
        best_fitness_oace = -float(pso_instance.optimizer.swarm.best_cost)
        
        print(f"\n✅ GA-PSO concluído!")
        print(f"   • Melhor posição encontrada: {best_pos}")
        print(f"   • Melhor score OACE: {best_fitness_oace:.6f}")
        
        # Garante que o melhor fitness está dentro do range válido [0, 1]
        if best_fitness_oace > 1.0:
            print(f"⚠️  AVISO: Score OACE inválido ({best_fitness_oace:.6f}) > 1.0. Corrigindo...")
            best_metrics = self._warm_up_candidate(best_pos)
            corrected_fitness = self._calculate_oace_score(best_metrics)
            best_fitness_oace = corrected_fitness
            print(f"   • Score OACE corrigido: {best_fitness_oace:.6f}")
        elif best_fitness_oace < 0.0:
            print(f"⚠️  AVISO: Score OACE inválido ({best_fitness_oace:.6f}) < 0.0. Corrigindo...")
            best_fitness_oace = 0.0
        
        best_metrics = self._warm_up_candidate(best_pos)
        pso_instance = getattr(self, 'pso_phase2', self.pso)
        final_population = pso_instance.optimizer.swarm.position
        final_fitness = -pso_instance.optimizer.swarm.pbest_cost
        self._print_population_summary(final_population, final_fitness, "GA-PSO Final")
        
        # Resumo final da Fase GA-PSO
        best_architecture, best_params = self._convert_to_architecture_params(best_pos)
        self._print_phase_summary("GA-PSO", best_fitness_oace, best_architecture, best_params)
        
        return best_pos, best_fitness_oace

    def _calculate_oace_score(self, metrics):
        """
        Calcula o score OACE para um conjunto de métricas.
        
        Args:
            metrics: Dicionário com as métricas
            
        Returns:
            float: Score OACE (entre 0 e 1)
        """
        print(f"   📊 Calculando score OACE...")

        assertiveness_weights, cost_weights, rc_a, rc_c = limited_scenario_weights()
        #assertiveness_weights, cost_weights, rc_a, rc_c = equilibrium_scenario_weights()
        #assertiveness_weights, cost_weights, rc_a, rc_c = critical_scenario_weights()

        self._update_metrics_ranges(metrics)
        
        print(f"      • Limites assertividade: {self.metrics_ranges['assertiveness']}")
        print(f"      • Limites custo: {self.metrics_ranges['cost']}")
        print(f"      • Lambda (trade-off): {self.lambda_param}")

        # Calcula o score OACE usando os limites atualizados
        score = calculate_oace_score(
            assertiveness_metrics={
                "top1_acc": metrics["top1_acc"], 
                "top5_acc": metrics["top5_acc"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            },
            cost_metrics={
                "total_params": metrics["total_params"],
                "avg_inference_time": metrics["avg_inference_time"],
                "memory_used_mb": metrics["memory_used_mb"],
                "gflops": metrics["gflops"],
            },
            lambda_param=self.lambda_param,
            assertiveness_weights=assertiveness_weights,
            cost_weights=cost_weights,
            assertiveness_min_max=self.metrics_ranges["assertiveness"],
            cost_min_max=self.metrics_ranges["cost"],
        )
        
        print(f"      • Score OACE calculado: {score:.6f}")
        
        # Validação: Garante que o score está no range correto
        if not (0.0 <= score <= 1.0):
            print(f"   ⚠️  AVISO: Score OACE fora do range [0,1]: {score:.6f}")
            print(f"   Métricas de assertividade:")
            for key in ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]:
                if key in metrics:
                    print(f"     {key}: {metrics[key]:.4f}")
            print(f"   Métricas de custo:")
            for key in ["total_params", "avg_inference_time", "memory_used_mb", "gflops"]:
                if key in metrics:
                    print(f"     {key}: {metrics[key]:.4f}")
            print(f"   Limites de assertividade: {self.metrics_ranges['assertiveness']}")
            print(f"   Limites de custo: {self.metrics_ranges['cost']}")
            score = max(0.0, min(1.0, score))
            print(f"   Score corrigido: {score:.6f}")
        
        return score

    def _update_metrics_ranges(self, new_metrics):
        """
        Atualiza os limites min/max das métricas dinamicamente para incluir novos valores.
        Isso evita que candidatos fiquem fora do range e gerem scores negativos.
        
        Args:
            new_metrics: Dicionário com novas métricas a serem incluídas nos limites
        """
        if self.metrics_ranges is None:
            print("   🔧 Inicializando limites das métricas com valores padrão...")
            self.metrics_ranges = {
                "assertiveness": {
                    "top1_acc": {"min": new_metrics.get("top1_acc", 0.0), "max": new_metrics.get("top1_acc", 1.0)},
                    "top5_acc": {"min": new_metrics.get("top5_acc", 0.0), "max": new_metrics.get("top5_acc", 1.0)},
                    "precision_macro": {"min": new_metrics.get("precision_macro", 0.0), "max": new_metrics.get("precision_macro", 1.0)},
                    "recall_macro": {"min": new_metrics.get("recall_macro", 0.0), "max": new_metrics.get("recall_macro", 1.0)},
                    "f1_macro": {"min": new_metrics.get("f1_macro", 0.0), "max": new_metrics.get("f1_macro", 1.0)},
                },
                "cost": {
                    "total_params": {"min": new_metrics.get("total_params", 0), "max": new_metrics.get("total_params", 1000000)},
                    "avg_inference_time": {"min": new_metrics.get("avg_inference_time", 0.0), "max": new_metrics.get("avg_inference_time", 1.0)},
                    "memory_used_mb": {"min": new_metrics.get("memory_used_mb", 0.0), "max": new_metrics.get("memory_used_mb", 1000.0)},
                    "gflops": {"min": new_metrics.get("gflops", 0.0), "max": new_metrics.get("gflops", 100.0)},
                }
            }
            print(f"   ✅ Limites inicializados com valores atuais")
            return
            
        # Atualiza limites de assertividade
        assertiveness_metrics = ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]
        for metric in assertiveness_metrics:
            if metric in new_metrics and metric in self.metrics_ranges["assertiveness"]:
                current_min = self.metrics_ranges["assertiveness"][metric]["min"]
                current_max = self.metrics_ranges["assertiveness"][metric]["max"]
                new_value = new_metrics[metric]
                
                # Atualiza min/max se necessário
                old_min, old_max = current_min, current_max
                self.metrics_ranges["assertiveness"][metric]["min"] = min(current_min, new_value)
                self.metrics_ranges["assertiveness"][metric]["max"] = max(current_max, new_value)
                
                # Log da atualização se houve mudança
                if old_min != self.metrics_ranges["assertiveness"][metric]["min"] or old_max != self.metrics_ranges["assertiveness"][metric]["max"]:
                    print(f"   📊 Atualizado limite {metric}: [{old_min:.4f}, {old_max:.4f}] → [{self.metrics_ranges['assertiveness'][metric]['min']:.4f}, {self.metrics_ranges['assertiveness'][metric]['max']:.4f}]")
        
        
        # Atualiza limites de custo
        cost_metrics = ["total_params", "avg_inference_time", "memory_used_mb", "gflops"]
        for metric in cost_metrics:
            if metric in new_metrics and metric in self.metrics_ranges["cost"]:
                current_min = self.metrics_ranges["cost"][metric]["min"]
                current_max = self.metrics_ranges["cost"][metric]["max"]
                new_value = new_metrics[metric]
                
                # Atualiza min/max se necessário
                old_min, old_max = current_min, current_max
                self.metrics_ranges["cost"][metric]["min"] = min(current_min, new_value)
                self.metrics_ranges["cost"][metric]["max"] = max(current_max, new_value)
                
                # Log da atualização se houve mudança
                if old_min != self.metrics_ranges["cost"][metric]["min"] or old_max != self.metrics_ranges["cost"][metric]["max"]:
                    print(f"   📊 Atualizado limite {metric}: [{old_min:.4f}, {old_max:.4f}] → [{self.metrics_ranges['cost'][metric]['min']:.4f}, {self.metrics_ranges['cost'][metric]['max']:.4f}]")



