"""
Gerenciador de Checkpoints para o algoritmo AFSA-GA-PSO.
Permite salvar e restaurar o estado da otimização para retomada após interrupções.


1: GPU_ID=3 nohup python3 -X utf8 -u -m main_checkpoint > results_checkpoint.log 2>&1 & disown
2 (checkpoint): GPU_ID=3 RESUME=1 nohup python3 -X utf8 -u -m main_checkpoint > results_checkpoint.log 2>&1 & disown
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib


class CheckpointManager:
    """
    Gerencia checkpoints para o algoritmo de otimização AFSA-GA-PSO.
    
    Salva o estado completo do algoritmo em pontos estratégicos para permitir
    retomada em caso de interrupção (crash, timeout, etc).
    
    Atributos:
        checkpoint_dir (str): Diretório onde os checkpoints são salvos
        experiment_id (str): Identificador único do experimento
        max_checkpoints (int): Número máximo de checkpoints a manter
        checkpoint_interval (int): Intervalo de iterações entre checkpoints
    """
    
    # Fases do algoritmo
    PHASE_INIT = "INIT"
    PHASE_AFSA_CANDIDATES = "AFSA_CANDIDATES_COMPLETE"
    PHASE_AFSA_PSO = "AFSA_PSO"
    PHASE_GA_PSO = "GA_PSO"
    PHASE_COMPLETE = "COMPLETE"
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        experiment_id: str = None,
        max_checkpoints: int = 3,
        checkpoint_interval: int = 1,
        verbose: bool = True
    ):
        """
        Inicializa o gerenciador de checkpoints.
        
        Args:
            checkpoint_dir: Diretório para salvar checkpoints
            experiment_id: ID único do experimento (gerado automaticamente se None)
            max_checkpoints: Número máximo de checkpoints a manter (remove os mais antigos)
            checkpoint_interval: Salvar checkpoint a cada N iterações
            verbose: Se True, imprime mensagens de status
        """
        self.checkpoint_dir = checkpoint_dir
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        
        # Cria diretório se não existir
        self._ensure_dir()
        
        if self.verbose:
            print(f"💾 CheckpointManager inicializado:")
            print(f"   • Diretório: {self.checkpoint_dir}")
            print(f"   • Experiment ID: {self.experiment_id}")
            print(f"   • Max checkpoints: {self.max_checkpoints}")
            print(f"   • Intervalo: a cada {self.checkpoint_interval} iteração(ões)")
    
    def _generate_experiment_id(self) -> str:
        """Gera um ID único para o experimento baseado no timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"
    
    def _ensure_dir(self):
        """Garante que o diretório de checkpoints existe."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_checkpoint_path(self, phase: str, iteration: int = None) -> str:
        """
        Gera o caminho do arquivo de checkpoint.
        
        Args:
            phase: Fase atual do algoritmo
            iteration: Número da iteração (opcional)
            
        Returns:
            Caminho completo do arquivo de checkpoint
        """
        if iteration is not None:
            filename = f"{self.experiment_id}_{phase}_iter{iteration:03d}.ckpt"
        else:
            filename = f"{self.experiment_id}_{phase}.ckpt"
        return os.path.join(self.checkpoint_dir, filename)
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Gera um hash da configuração para validação.
        
        Args:
            config: Dicionário de configuração do algoritmo
            
        Returns:
            Hash MD5 da configuração
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_checkpoint(
        self,
        phase: str,
        iteration: int,
        state: Dict[str, Any],
        config: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Salva um checkpoint com o estado atual do algoritmo.
        
        Args:
            phase: Fase atual (AFSA_PSO, GA_PSO, etc)
            iteration: Número da iteração atual
            state: Dicionário com todo o estado a ser salvo
            config: Configuração do algoritmo (para validação na retomada)
            metadata: Metadados adicionais (opcional)
            
        Returns:
            Caminho do checkpoint salvo
        """
        checkpoint_path = self._get_checkpoint_path(phase, iteration)
        
        checkpoint = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "phase": phase,
            "iteration": iteration,
            "config_hash": self._get_config_hash(config),
            "config": config,
            "state": state,
            "metadata": metadata or {}
        }
        
        # Salva o checkpoint
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.verbose:
                print(f"\n💾 Checkpoint salvo: {os.path.basename(checkpoint_path)}")
                print(f"   • Fase: {phase}, Iteração: {iteration}")
                print(f"   • Tamanho: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")
            
            # Limpa checkpoints antigos
            self._cleanup_old_checkpoints(phase)
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str = None) -> Optional[Dict[str, Any]]:
        """
        Carrega um checkpoint existente.
        
        Args:
            checkpoint_path: Caminho específico do checkpoint (se None, carrega o mais recente)
            
        Returns:
            Dicionário com o checkpoint ou None se não encontrar
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            if self.verbose:
                print("ℹ️  Nenhum checkpoint encontrado para retomar.")
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if self.verbose:
                print(f"\n📂 Checkpoint carregado: {os.path.basename(checkpoint_path)}")
                print(f"   • Fase: {checkpoint['phase']}, Iteração: {checkpoint['iteration']}")
                print(f"   • Salvo em: {checkpoint['timestamp']}")
                print(f"   • Experiment ID: {checkpoint['experiment_id']}")
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ Erro ao carregar checkpoint: {e}")
            return None
    
    def validate_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Valida se um checkpoint é compatível com a configuração atual.
        
        Args:
            checkpoint: Checkpoint carregado
            current_config: Configuração atual do algoritmo
            
        Returns:
            Tuple (é_válido, mensagem)
        """
        # Verifica versão
        if checkpoint.get("version") != "1.0":
            return False, f"Versão incompatível: {checkpoint.get('version')}"
        
        # Verifica hash da configuração
        current_hash = self._get_config_hash(current_config)
        saved_hash = checkpoint.get("config_hash")
        
        if current_hash != saved_hash:
            # Identifica diferenças
            saved_config = checkpoint.get("config", {})
            differences = []
            
            for key in set(list(current_config.keys()) + list(saved_config.keys())):
                if str(current_config.get(key)) != str(saved_config.get(key)):
                    differences.append(f"{key}: {saved_config.get(key)} → {current_config.get(key)}")
            
            return False, f"Configuração diferente:\n   " + "\n   ".join(differences)
        
        return True, "Checkpoint válido"
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Encontra o checkpoint mais recente para o experimento atual.
        
        Returns:
            Caminho do checkpoint mais recente ou None
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        # Lista todos os checkpoints do experimento atual
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(self.experiment_id) and file.endswith('.ckpt'):
                path = os.path.join(self.checkpoint_dir, file)
                checkpoints.append((path, os.path.getmtime(path)))
        
        if not checkpoints:
            return None
        
        # Retorna o mais recente
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    def find_any_checkpoint(self) -> Optional[str]:
        """
        Encontra qualquer checkpoint disponível no diretório.
        
        Returns:
            Caminho do checkpoint mais recente ou None
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.ckpt'):
                path = os.path.join(self.checkpoint_dir, file)
                checkpoints.append((path, os.path.getmtime(path)))
        
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Lista todos os checkpoints disponíveis.
        
        Returns:
            Lista de dicionários com informações dos checkpoints
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.ckpt'):
                path = os.path.join(self.checkpoint_dir, file)
                try:
                    with open(path, 'rb') as f:
                        ckpt = pickle.load(f)
                    checkpoints.append({
                        "path": path,
                        "filename": file,
                        "experiment_id": ckpt.get("experiment_id"),
                        "phase": ckpt.get("phase"),
                        "iteration": ckpt.get("iteration"),
                        "timestamp": ckpt.get("timestamp"),
                        "size_kb": os.path.getsize(path) / 1024
                    })
                except:
                    pass
        
        return sorted(checkpoints, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def _cleanup_old_checkpoints(self, phase: str):
        """
        Remove checkpoints antigos mantendo apenas os mais recentes.
        
        Args:
            phase: Fase atual (para limpar apenas checkpoints da mesma fase)
        """
        if not os.path.exists(self.checkpoint_dir):
            return
        
        # Lista checkpoints da fase atual
        phase_checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(self.experiment_id) and phase in file and file.endswith('.ckpt'):
                path = os.path.join(self.checkpoint_dir, file)
                phase_checkpoints.append((path, os.path.getmtime(path)))
        
        # Ordena por data (mais recente primeiro)
        phase_checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove os mais antigos
        for path, _ in phase_checkpoints[self.max_checkpoints:]:
            try:
                os.remove(path)
                if self.verbose:
                    print(f"   🗑️  Checkpoint antigo removido: {os.path.basename(path)}")
            except:
                pass
    
    def should_save_checkpoint(self, iteration: int) -> bool:
        """
        Verifica se deve salvar checkpoint na iteração atual.
        
        Args:
            iteration: Número da iteração atual
            
        Returns:
            True se deve salvar, False caso contrário
        """
        return iteration % self.checkpoint_interval == 0
    
    def delete_experiment_checkpoints(self):
        """Remove todos os checkpoints do experimento atual."""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(self.experiment_id) and file.endswith('.ckpt'):
                try:
                    os.remove(os.path.join(self.checkpoint_dir, file))
                    if self.verbose:
                        print(f"   🗑️  Removido: {file}")
                except:
                    pass


def extract_pso_state(pso_instance) -> Dict[str, Any]:
    """
    Extrai o estado completo de uma instância PSO.
    
    Args:
        pso_instance: Instância do PSO
        
    Returns:
        Dicionário com o estado do PSO
    """
    swarm = pso_instance.optimizer.swarm
    return {
        "position": swarm.position.copy() if isinstance(swarm.position, np.ndarray) else np.array(swarm.position),
        "velocity": swarm.velocity.copy() if hasattr(swarm, 'velocity') and swarm.velocity is not None else None,
        "pbest_pos": swarm.pbest_pos.copy() if isinstance(swarm.pbest_pos, np.ndarray) else np.array(swarm.pbest_pos),
        "pbest_cost": swarm.pbest_cost.copy() if isinstance(swarm.pbest_cost, np.ndarray) else np.array(swarm.pbest_cost),
        "best_pos": swarm.best_pos.copy() if isinstance(swarm.best_pos, np.ndarray) else np.array(swarm.best_pos),
        "best_cost": float(swarm.best_cost),
    }


def restore_pso_state(pso_instance, state: Dict[str, Any]):
    """
    Restaura o estado de uma instância PSO.
    
    Args:
        pso_instance: Instância do PSO
        state: Estado salvo anteriormente
    """
    swarm = pso_instance.optimizer.swarm
    swarm.position = state["position"].copy()
    if state["velocity"] is not None and hasattr(swarm, 'velocity'):
        swarm.velocity = state["velocity"].copy()
    swarm.pbest_pos = state["pbest_pos"].copy()
    swarm.pbest_cost = state["pbest_cost"].copy()
    swarm.best_pos = state["best_pos"].copy()
    swarm.best_cost = state["best_cost"]
