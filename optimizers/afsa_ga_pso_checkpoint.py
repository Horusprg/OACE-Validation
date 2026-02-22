"""
Wrapper do AFSA-GA-PSO com suporte a checkpoint.
Permite retomar a otimização de onde parou em caso de interrupção.

Uso:
    from optimizers.afsa_ga_pso_checkpoint import AFSAGAPSOWithCheckpoint
    
    optimizer = AFSAGAPSOWithCheckpoint(
        population_size=20,
        max_iter=10,
        # ... outros parâmetros ...
        checkpoint_dir="checkpoints",
        checkpoint_interval=1,
        resume_from_checkpoint=True  # ou caminho específico
    )
    
    best_architecture, best_params, best_fitness = optimizer.optimize()
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.afsa_ga_pso import AFSAGAPSO
from optimizers.pso import PSO
from utils.checkpoint_manager import CheckpointManager, extract_pso_state, restore_pso_state


class AFSAGAPSOWithCheckpoint(AFSAGAPSO):
    """
    Extensão do AFSA-GA-PSO com suporte a checkpoint para retomada.
    
    Herda toda a funcionalidade do AFSAGAPSO original e adiciona:
    - Salvamento automático de checkpoints durante a otimização
    - Retomada de onde parou em caso de interrupção
    - Validação de compatibilidade de checkpoints
    
    Atributos adicionais:
        checkpoint_manager: Gerenciador de checkpoints
        resume_from_checkpoint: Se deve tentar retomar de checkpoint existente
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
        device = None,
        # Parâmetros de checkpoint
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 1,
        resume_from_checkpoint: Union[bool, str] = False,
        experiment_id: str = None,
        max_checkpoints: int = 3
    ):
        """
        Inicializa o otimizador com suporte a checkpoint.
        
        Args:
            ... (mesmos parâmetros do AFSAGAPSO) ...
            checkpoint_dir: Diretório para salvar checkpoints
            checkpoint_interval: Salvar checkpoint a cada N iterações
            resume_from_checkpoint: True para retomar automaticamente, 
                                   ou caminho específico do checkpoint
            experiment_id: ID único do experimento (gerado automaticamente se None)
            max_checkpoints: Número máximo de checkpoints a manter por fase
        """
        # Inicializa a classe base
        super().__init__(
            population_size=population_size,
            max_iter=max_iter,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=classes,
            lambda_param=lambda_param,
            afsa_params=afsa_params,
            pso_params=pso_params,
            ga_params=ga_params,
            architectures_to_optimize=architectures_to_optimize,
            log_dir=log_dir,
            device=device
        )
        
        # Configuração de checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Inicializa o gerenciador de checkpoints
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_id=experiment_id,
            max_checkpoints=max_checkpoints,
            checkpoint_interval=checkpoint_interval,
            verbose=True
        )
        
        # Guarda a configuração para validação
        self._config = {
            "population_size": population_size,
            "max_iter": max_iter,
            "lambda_param": lambda_param,
            "afsa_params": afsa_params or {},
            "pso_params": pso_params or {},
            "ga_params": ga_params or {},
            "architectures_to_optimize": architectures_to_optimize or [],
            "n_dim": self.n_dim,
            "param_bounds": {k: list(v) for k, v in self.param_bounds.items()}
        }
        
        print(f"\n🔄 AFSAGAPSOWithCheckpoint inicializado")
        print(f"   • Checkpoint dir: {checkpoint_dir}")
        print(f"   • Checkpoint interval: {checkpoint_interval}")
        print(f"   • Resume: {resume_from_checkpoint}")
    
    def _build_state_dict(
        self,
        phase: str,
        iteration: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Constrói o dicionário de estado para checkpoint.
        
        Args:
            phase: Fase atual do algoritmo
            iteration: Iteração atual
            **kwargs: Dados adicionais específicos da fase
            
        Returns:
            Dicionário com todo o estado necessário para retomada
        """
        state = {
            "phase": phase,
            "iteration": iteration,
            "best_solution": self.best_solution.tolist() if self.best_solution is not None else None,
            "best_fitness": self.best_fitness,
            "candidates_cache": {str(k): v for k, v in self.candidates_cache.items()},
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "metrics_ranges": self.metrics_ranges,
            "metrics_history": [
                (c.tolist() if isinstance(c, np.ndarray) else c, m) 
                for c, m in self.metrics_history
            ] if self.metrics_history else [],
        }
        
        # Adiciona dados específicos
        state.update(kwargs)
        
        return state
    
    def _restore_base_state(self, state: Dict[str, Any]):
        """
        Restaura o estado base do otimizador.
        
        Args:
            state: Estado salvo no checkpoint
        """
        # Restaura cache
        if state.get("candidates_cache"):
            self.candidates_cache = {
                eval(k) if k.startswith("(") else k: v 
                for k, v in state["candidates_cache"].items()
            }
        
        self.cache_hits = state.get("cache_hits", 0)
        self.cache_misses = state.get("cache_misses", 0)
        
        # Restaura métricas
        self.metrics_ranges = state.get("metrics_ranges")
        
        if state.get("metrics_history"):
            self.metrics_history = [
                (np.array(c) if isinstance(c, list) else c, m)
                for c, m in state["metrics_history"]
            ]
        
        # Restaura melhores resultados
        if state.get("best_solution") is not None:
            self.best_solution = np.array(state["best_solution"])
        self.best_fitness = state.get("best_fitness", float("-inf"))
        
        print(f"   ✓ Estado base restaurado")
        print(f"     • Cache: {len(self.candidates_cache)} entradas")
        print(f"     • Best fitness: {self.best_fitness:.6f}")
    
    def optimize(self):
        """
        Executa o processo de otimização híbrida AFSA-GA-PSO com suporte a checkpoint.
        
        Sobrescreve o método da classe base para adicionar:
        - Verificação de checkpoint existente para retomada
        - Salvamento de checkpoints em pontos estratégicos
        
        Returns:
            tuple: (melhor arquitetura encontrada, parâmetros, melhor valor de fitness)
        """
        self._print_header("INICIANDO OTIMIZAÇÃO HÍBRIDA AFSA-GA-PSO (COM CHECKPOINT)")
        self._print_configuration()
        
        # Verifica se deve retomar de checkpoint
        loaded_checkpoint = None
        if self.resume_from_checkpoint:
            if isinstance(self.resume_from_checkpoint, str):
                # Caminho específico fornecido
                loaded_checkpoint = self.checkpoint_manager.load_checkpoint(self.resume_from_checkpoint)
            else:
                # Busca o checkpoint mais recente
                latest = self.checkpoint_manager.find_any_checkpoint()
                if latest:
                    loaded_checkpoint = self.checkpoint_manager.load_checkpoint(latest)
        
        # Valida e decide se vai retomar
        resume_phase = None
        resume_iteration = 0
        resume_data = {}
        
        if loaded_checkpoint:
            is_valid, msg = self.checkpoint_manager.validate_checkpoint(
                loaded_checkpoint, self._config
            )
            
            if is_valid:
                print(f"\n✅ Checkpoint válido! Retomando execução...")
                resume_phase = loaded_checkpoint["state"]["phase"]
                resume_iteration = loaded_checkpoint["state"]["iteration"]
                resume_data = loaded_checkpoint["state"]
                
                # Restaura estado base
                self._restore_base_state(resume_data)
                
                # Atualiza experiment_id para continuar no mesmo experimento
                self.checkpoint_manager.experiment_id = loaded_checkpoint["experiment_id"]
            else:
                print(f"\n⚠️  Checkpoint inválido, iniciando do zero.")
                print(f"   Motivo: {msg}")
        
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
            
            # ====== FASE 1: OTIMIZAÇÃO INICIAL COM AFSA-PSO ======
            
            # Verifica se deve pular a geração de candidatos (retomada)
            if resume_phase in [CheckpointManager.PHASE_AFSA_PSO, CheckpointManager.PHASE_GA_PSO]:
                # Retomando de fase posterior - restaurar candidatos
                print(f"\n🔄 Retomando: pulando geração de candidatos iniciais")
                initial_population = np.array(resume_data["initial_population"])
                candidates_metrics = [
                    (np.array(c), m) for c, m in resume_data["candidates_metrics"]
                ]
            else:
                # Execução normal ou retomada da fase inicial
                self._print_section("FASE 1: OTIMIZAÇÃO INICIAL COM AFSA-PSO")
                
                # Gera população inicial usando AFSA
                self._print_step("Gerando população inicial diversificada com AFSA",
                               f"Tamanho: {self.population_size}, Iterações: {self.afsa_params['max_iter']}")
                initial_population, candidates_metrics = self._generate_initial_candidates()
                
                # Calcula limites das métricas
                self._print_step("Calculando limites das métricas para normalização OACE")
                self._calculate_metrics_ranges(candidates_metrics)
                
                # 💾 CHECKPOINT: Após geração de candidatos
                self._save_candidates_checkpoint(initial_population, candidates_metrics)
            
            # Executa AFSA-PSO (com possível retomada)
            if resume_phase == CheckpointManager.PHASE_GA_PSO:
                # Pular diretamente para GA-PSO
                print(f"\n🔄 Retomando: pulando fase AFSA-PSO")
                phase1_solutions = np.array(resume_data["phase1_solutions"])
            else:
                # Executa AFSA-PSO
                phase1_solutions = self._execute_afsa_pso_phase_with_checkpoint(
                    initial_population, 
                    candidates_metrics,
                    resume_phase=resume_phase,
                    resume_iteration=resume_iteration if resume_phase == CheckpointManager.PHASE_AFSA_PSO else 0,
                    resume_data=resume_data if resume_phase == CheckpointManager.PHASE_AFSA_PSO else {}
                )
            
            best_idx = np.argmax([self.fitness_function(x) for x in phase1_solutions])
            best_arch, best_params = self._convert_to_architecture_params(phase1_solutions[best_idx])
            best_fitness = self.fitness_function(phase1_solutions[best_idx])
            self._print_phase_summary("AFSA-PSO", best_fitness, best_arch, best_params)
            
            # ====== FASE 2: OTIMIZAÇÃO GLOBAL COM GA-PSO ======
            self._print_section("FASE 2: OTIMIZAÇÃO GLOBAL COM GA-PSO")
            
            # Executa GA-PSO (com possível retomada)
            best_position, best_fitness = self._execute_ga_pso_phase_with_checkpoint(
                phase1_solutions,
                resume_phase=resume_phase,
                resume_iteration=resume_iteration if resume_phase == CheckpointManager.PHASE_GA_PSO else 0,
                resume_data=resume_data if resume_phase == CheckpointManager.PHASE_GA_PSO else {}
            )
            
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
            
            # 💾 CHECKPOINT FINAL: Marca conclusão
            self._save_final_checkpoint(best_architecture_name, best_architecture_params, 
                                       best_fitness, final_metrics)
            
            return best_architecture_name, best_architecture_params, best_fitness
            
        except Exception as e:
            print(f"\n❌ Erro durante a otimização: {str(e)}")
            print(f"💾 Tentando salvar checkpoint de emergência...")
            
            # Tenta salvar checkpoint de emergência
            try:
                self._save_emergency_checkpoint(str(e))
            except:
                print("   ⚠️  Não foi possível salvar checkpoint de emergência")
            
            if hasattr(self, 'logger'):
                try:
                    self.logger._save_log()
                    print(f"\n✓ Logs parciais salvos em: {self.logger.log_dir}/{self.logger.current_experiment}")
                except:
                    pass
            raise
    
    def _save_candidates_checkpoint(
        self, 
        initial_population: np.ndarray, 
        candidates_metrics: List[Tuple[np.ndarray, Dict[str, float]]]
    ):
        """Salva checkpoint após geração de candidatos iniciais."""
        state = self._build_state_dict(
            phase=CheckpointManager.PHASE_AFSA_CANDIDATES,
            iteration=0,
            initial_population=initial_population.tolist(),
            candidates_metrics=[
                (c.tolist() if isinstance(c, np.ndarray) else c, m) 
                for c, m in candidates_metrics
            ]
        )
        
        self.checkpoint_manager.save_checkpoint(
            phase=CheckpointManager.PHASE_AFSA_CANDIDATES,
            iteration=0,
            state=state,
            config=self._config,
            metadata={"description": "Após geração de candidatos iniciais"}
        )
    
    def _execute_afsa_pso_phase_with_checkpoint(
        self,
        initial_population: np.ndarray,
        candidates_metrics: List[Tuple[np.ndarray, Dict[str, float]]],
        resume_phase: str = None,
        resume_iteration: int = 0,
        resume_data: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Executa a Fase 1: AFSA-PSO com suporte a checkpoint.
        
        Args:
            initial_population: População inicial
            candidates_metrics: Métricas dos candidatos
            resume_phase: Fase de retomada (se houver)
            resume_iteration: Iteração de retomada (se houver)
            resume_data: Dados de retomada (se houver)
            
        Returns:
            Soluções da Fase 1
        """
        self._print_section("AFSA-PSO: Iniciando Fase 1 com otimização integrada")
        
        # Determina iteração inicial
        start_iteration = 0
        if resume_phase == CheckpointManager.PHASE_AFSA_PSO:
            start_iteration = resume_iteration + 1
            print(f"\n🔄 Retomando AFSA-PSO da iteração {start_iteration}")
        
        # Calcula fitness inicial dos candidatos
        if start_iteration == 0:
            self._print_step("Calculando fitness dos candidatos iniciais com OACE")
            initial_fitness = []
            for i, (candidate, metrics) in enumerate(candidates_metrics, 1):
                print(f"\n🔄 Avaliando candidato {i}/{len(candidates_metrics)}")
                score = self._calculate_oace_score(metrics)
                initial_fitness.append(score)
                print(f"   🎯 Score OACE: {score:.6f}")
            
            initial_fitness = np.array(initial_fitness)
        else:
            # Restaurar fitness do checkpoint
            initial_fitness = np.array(resume_data.get("initial_fitness", []))
        
        best_idx = np.argmax(initial_fitness)
        self._print_population_summary(initial_population, initial_fitness, "AFSA-PSO Inicial")
        
        print(f"\n🏆 Melhor candidato inicial:")
        print(f"   • Índice: {best_idx}")
        print(f"   • Score OACE: {initial_fitness[best_idx]:.6f}")
        arch_name, arch_params = self._convert_to_architecture_params(initial_population[best_idx])
        print(f"   • Arquitetura: {arch_name}")
        print(f"   • Parâmetros: {arch_params}")
        
        if start_iteration == 0:
            self.logger.log_iteration(
                iteration=0,
                phase="AFSA-PSO",
                population=initial_population,
                fitness_values=initial_fitness,
                best_position=initial_population[best_idx],
                best_fitness=initial_fitness[best_idx],
                metrics=candidates_metrics[best_idx][1],
                oace_score=initial_fitness[best_idx],
                pbest_pos=initial_population,
                pbest_cost=initial_fitness,
                gbest_pos=initial_population[best_idx],
                gbest_cost=initial_fitness[best_idx]
            )
        
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
        
        self.pso.fitness_function = pso_fitness_function
        
        # Inicializa ou restaura PSO
        if resume_phase == CheckpointManager.PHASE_AFSA_PSO and resume_data.get("pso_state"):
            print(f"   🔄 Restaurando estado do PSO...")
            self.pso.initialize_swarm_with_population(initial_population)
            restore_pso_state(self.pso, resume_data["pso_state"])
        else:
            self._print_step("Inicializando enxame PSO com população inicial")
            self.pso.initialize_swarm_with_population(initial_population)
        
        # Configura AFSA
        if self.afsa is None:
            from optimizers.afsa import AFSA
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
        
        self.afsa.fitness_function = self.fitness_function
        
        # Loop AFSA-PSO
        max_afsa_iter = self.afsa_params.get('max_iter', 3)
        self._print_step("Iniciando loop alternado AFSA-PSO",
                        f"Iterações AFSA: {max_afsa_iter}, Iterações PSO por ciclo: 1")
        
        for afsa_iter in range(start_iteration, max_afsa_iter):
            print(f"\n🔄 AFSA-PSO Iteração {afsa_iter + 1}/{max_afsa_iter}")
            
            # Aplica comportamentos AFSA
            self._print_step(f"AFSA aplicando comportamentos (iteração {afsa_iter + 1})")
            pso_gbest_before_afsa = -float(self.pso.optimizer.swarm.best_cost)
            
            print(f"\n      🔄 PSO ANTES DO AFSA:")
            print(f"         • Melhor OACE (gbest): {pso_gbest_before_afsa:.6f}")
            
            afsa_optimized_population = self._apply_afsa_behaviors_to_pso(afsa_iter)
            
            # Atualiza PSO
            self.pso.optimizer.swarm.position = afsa_optimized_population.copy()
            fitness_values = self.pso.fitness_function(afsa_optimized_population)
            
            # Atualiza pbest
            pbest_updated = 0
            for i in range(len(afsa_optimized_population)):
                if fitness_values[i] < self.pso.optimizer.swarm.pbest_cost[i]:
                    self.pso.optimizer.swarm.pbest_pos[i] = afsa_optimized_population[i].copy()
                    self.pso.optimizer.swarm.pbest_cost[i] = fitness_values[i]
                    pbest_updated += 1
            
            # Atualiza gbest
            best_idx = np.argmin(self.pso.optimizer.swarm.pbest_cost)
            new_gbest_cost = self.pso.optimizer.swarm.pbest_cost[best_idx]
            
            if new_gbest_cost < self.pso.optimizer.swarm.best_cost:
                old_gbest = -float(self.pso.optimizer.swarm.best_cost)
                self.pso.optimizer.swarm.best_pos = self.pso.optimizer.swarm.pbest_pos[best_idx].copy()
                self.pso.optimizer.swarm.best_cost = new_gbest_cost
                new_gbest = -float(self.pso.optimizer.swarm.best_cost)
                print(f"         • ✅ gbest ATUALIZADO: {old_gbest:.6f} → {new_gbest:.6f}")
            
            # PSO refina
            try:
                self.pso._update_swarm_one_iteration()
            except Exception as e:
                print(f"         • ⚠️  Erro ao executar PSO: {e}")
            
            pso_gbest_after = -float(self.pso.optimizer.swarm.best_cost)
            print(f"\n      📈 RESUMO: OACE {pso_gbest_before_afsa:.6f} → {pso_gbest_after:.6f}")
            
            # Log da iteração
            current_population = self.pso.optimizer.swarm.position
            current_fitness = np.array([self.fitness_function(p) for p in current_population])
            best_pos_afsa = self.pso.optimizer.swarm.best_pos
            
            try:
                best_metrics_afsa = self._warm_up_candidate(best_pos_afsa)
            except:
                best_metrics_afsa = None
            
            self.logger.log_iteration(
                iteration=afsa_iter + 1,
                phase="AFSA-PSO",
                population=current_population,
                fitness_values=current_fitness,
                best_position=best_pos_afsa,
                best_fitness=-float(self.pso.optimizer.swarm.best_cost),
                metrics=best_metrics_afsa,
                oace_score=-float(self.pso.optimizer.swarm.best_cost),
                pbest_pos=self.pso.optimizer.swarm.pbest_pos,
                pbest_cost=-self.pso.optimizer.swarm.pbest_cost,
                gbest_pos=self.pso.optimizer.swarm.best_pos,
                gbest_cost=-float(self.pso.optimizer.swarm.best_cost)
            )
            
            # 💾 CHECKPOINT: Salva a cada intervalo
            if self.checkpoint_manager.should_save_checkpoint(afsa_iter + 1):
                state = self._build_state_dict(
                    phase=CheckpointManager.PHASE_AFSA_PSO,
                    iteration=afsa_iter,
                    initial_population=initial_population.tolist(),
                    candidates_metrics=[
                        (c.tolist() if isinstance(c, np.ndarray) else c, m)
                        for c, m in candidates_metrics
                    ],
                    initial_fitness=initial_fitness.tolist(),
                    pso_state=extract_pso_state(self.pso)
                )
                
                self.checkpoint_manager.save_checkpoint(
                    phase=CheckpointManager.PHASE_AFSA_PSO,
                    iteration=afsa_iter,
                    state=state,
                    config=self._config,
                    metadata={"gbest": pso_gbest_after}
                )
        
        # Retorna soluções
        phase1_solutions = self.pso.optimizer.swarm.pbest_pos.copy()
        final_fitness = np.array([self.fitness_function(p) for p in phase1_solutions])
        
        self._print_population_summary(phase1_solutions, final_fitness, "AFSA-PSO Final")
        
        print(f"\n✅ Fase AFSA-PSO Concluída!")
        print(f"   • Melhor score: {np.max(final_fitness):.6f}")
        
        return phase1_solutions
    
    def _execute_ga_pso_phase_with_checkpoint(
        self,
        phase1_solutions: np.ndarray,
        resume_phase: str = None,
        resume_iteration: int = 0,
        resume_data: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Executa a Fase 2: GA-PSO com suporte a checkpoint.
        
        Args:
            phase1_solutions: Soluções da Fase 1
            resume_phase: Fase de retomada (se houver)
            resume_iteration: Iteração de retomada (se houver)
            resume_data: Dados de retomada (se houver)
            
        Returns:
            Tuple (melhor posição, melhor fitness)
        """
        self._print_section("GA-PSO: Iniciando Fase 2 com soluções da Fase 1")
        
        # Determina iteração inicial
        start_iteration = 0
        if resume_phase == CheckpointManager.PHASE_GA_PSO:
            start_iteration = resume_iteration + 1
            print(f"\n🔄 Retomando GA-PSO da iteração {start_iteration}")
        
        # Configura função de fitness
        def pso_fitness_function(x):
            if x.ndim == 1:
                return -self.fitness_function(x)
            else:
                return -np.array([self.fitness_function(xi) for xi in x])
        
        # Inicializa PSO Fase 2
        self._print_step("Inicializando PSO para Fase 2 com soluções da Fase 1")
        pso_phase2 = PSO(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=1,
            lower_bound=0.0,
            upper_bound=1.0,
            afsa_params=None,
            pso_options=self.pso_params,
            logger=self.logger
        )
        pso_phase2.fitness_function = pso_fitness_function
        pso_phase2.initialize_swarm_with_population(phase1_solutions)
        self.pso_phase2 = pso_phase2
        
        # Restaura estado se retomando
        if resume_phase == CheckpointManager.PHASE_GA_PSO and resume_data.get("pso_phase2_state"):
            print(f"   🔄 Restaurando estado do PSO Fase 2...")
            restore_pso_state(pso_phase2, resume_data["pso_phase2_state"])
        
        # Avalia fitness inicial
        initial_fitness = np.array([self.fitness_function(x) for x in phase1_solutions])
        best_idx = np.argmax(initial_fitness)
        
        if start_iteration == 0:
            best_metrics = self._warm_up_candidate(phase1_solutions[best_idx])
            self._print_population_summary(phase1_solutions, initial_fitness, "GA-PSO Inicial")
            
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
        
        # Loop GA-PSO
        max_ga_iter = self.ga_params.get('max_iter', self.max_iter)
        self._print_step("Iniciando loop alternado GA-PSO",
                        f"Iterações GA: {max_ga_iter}, Iterações PSO por ciclo: 1")
        
        for ga_iter in range(start_iteration, max_ga_iter):
            print(f"\n🔄 GA-PSO Iteração {ga_iter + 1}/{max_ga_iter}")
            
            pso_instance = self.pso_phase2
            pso_gbest_before_ga = -float(pso_instance.optimizer.swarm.best_cost)
            
            print(f"\n      🔄 PSO ANTES DO GA:")
            print(f"         • Melhor OACE (gbest): {pso_gbest_before_ga:.6f}")
            
            # GA aplica operadores
            self._print_step(f"GA aplicando operadores genéticos (iteração {ga_iter + 1})")
            ga_optimized_population = self._apply_ga_operators_to_pso(ga_iter)
            
            # Atualiza PSO
            pso_instance.optimizer.swarm.position = ga_optimized_population.copy()
            fitness_values = pso_instance.fitness_function(ga_optimized_population)
            
            # Atualiza pbest
            pbest_updates = 0
            for i in range(len(ga_optimized_population)):
                if fitness_values[i] < pso_instance.optimizer.swarm.pbest_cost[i]:
                    pso_instance.optimizer.swarm.pbest_pos[i] = ga_optimized_population[i].copy()
                    pso_instance.optimizer.swarm.pbest_cost[i] = fitness_values[i]
                    pbest_updates += 1
            
            # Atualiza gbest
            if pso_instance.optimizer.swarm.pbest_cost.size > 0:
                best_idx_pso = np.argmin(pso_instance.optimizer.swarm.pbest_cost)
                if pso_instance.optimizer.swarm.pbest_cost[best_idx_pso] < pso_instance.optimizer.swarm.best_cost:
                    old_gbest = -float(pso_instance.optimizer.swarm.best_cost)
                    pso_instance.optimizer.swarm.best_pos = pso_instance.optimizer.swarm.pbest_pos[best_idx_pso].copy()
                    pso_instance.optimizer.swarm.best_cost = pso_instance.optimizer.swarm.pbest_cost[best_idx_pso]
                    new_gbest = -float(pso_instance.optimizer.swarm.best_cost)
                    print(f"         • ✅ gbest ATUALIZADO: {old_gbest:.6f} → {new_gbest:.6f}")
            
            # PSO refina
            self._print_step("PSO refinando soluções modificadas pelo GA")
            try:
                pso_instance._update_swarm_one_iteration()
            except Exception as e:
                print(f"         ⚠️  Erro na iteração do PSO: {e}")
            
            pso_gbest_after = -float(pso_instance.optimizer.swarm.best_cost)
            print(f"\n      📈 RESUMO: OACE {pso_gbest_before_ga:.6f} → {pso_gbest_after:.6f}")
            
            # Logging
            current_oace_fitness = -pso_instance.optimizer.swarm.pbest_cost
            best_oace_score = -float(pso_instance.optimizer.swarm.best_cost)
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
            
            # 💾 CHECKPOINT: Salva a cada intervalo
            if self.checkpoint_manager.should_save_checkpoint(ga_iter + 1):
                state = self._build_state_dict(
                    phase=CheckpointManager.PHASE_GA_PSO,
                    iteration=ga_iter,
                    phase1_solutions=phase1_solutions.tolist(),
                    pso_phase2_state=extract_pso_state(pso_instance)
                )
                
                self.checkpoint_manager.save_checkpoint(
                    phase=CheckpointManager.PHASE_GA_PSO,
                    iteration=ga_iter,
                    state=state,
                    config=self._config,
                    metadata={"gbest": pso_gbest_after}
                )
        
        # Retorna melhor solução
        best_pos = pso_instance.optimizer.swarm.best_pos.copy()
        best_fitness_oace = -float(pso_instance.optimizer.swarm.best_cost)
        
        print(f"\n✅ GA-PSO concluído!")
        print(f"   • Melhor score OACE: {best_fitness_oace:.6f}")
        
        # Garante range válido
        best_fitness_oace = max(0.0, min(1.0, best_fitness_oace))
        
        return best_pos, best_fitness_oace
    
    def _save_final_checkpoint(
        self,
        best_architecture: str,
        best_params: Dict[str, Any],
        best_fitness: float,
        final_metrics: Dict[str, float]
    ):
        """Salva checkpoint final marcando conclusão."""
        state = self._build_state_dict(
            phase=CheckpointManager.PHASE_COMPLETE,
            iteration=-1,
            best_architecture=best_architecture,
            best_params=best_params,
            final_metrics=final_metrics
        )
        
        self.checkpoint_manager.save_checkpoint(
            phase=CheckpointManager.PHASE_COMPLETE,
            iteration=0,
            state=state,
            config=self._config,
            metadata={
                "description": "Otimização concluída com sucesso",
                "best_fitness": best_fitness,
                "best_architecture": best_architecture
            }
        )
        
        print(f"\n✅ Checkpoint final salvo. Otimização concluída!")
    
    def _save_emergency_checkpoint(self, error_message: str):
        """Salva checkpoint de emergência em caso de erro."""
        state = self._build_state_dict(
            phase="EMERGENCY",
            iteration=-1,
            error_message=error_message
        )
        
        # Tenta salvar estados dos PSOs se disponíveis
        if hasattr(self, 'pso') and self.pso is not None:
            try:
                state["pso_state"] = extract_pso_state(self.pso)
            except:
                pass
        
        if hasattr(self, 'pso_phase2') and self.pso_phase2 is not None:
            try:
                state["pso_phase2_state"] = extract_pso_state(self.pso_phase2)
            except:
                pass
        
        self.checkpoint_manager.save_checkpoint(
            phase="EMERGENCY",
            iteration=0,
            state=state,
            config=self._config,
            metadata={"error": error_message}
        )
        
        print(f"💾 Checkpoint de emergência salvo!")
