#!/usr/bin/env python3
"""
Script de treinamento completo para ResNet no CIFAR-10.
Segue a lógica dos arquivos de treinamento do EfficientNet e MobileNet,
permitindo configurar hiperparâmetros e opções avançadas de treino.
"""

import os
import sys
import json
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import get_cifar10_dataloaders
from utils.training_utils import train_model, get_optimized_scheduler
from utils.evaluate_utils import evaluate_model
from models.ResNet.resnet_architecture import ResNetParams, generate_resnet_architecture


def train_resnet_full():
	"""
	Treina uma ResNet com configurações robustas e hiperparâmetros configuráveis.
	"""
	print("=" * 70)
	print("TREINAMENTO COMPLETO RESNET (CIFAR-10)")
	print("=" * 70)

	# Configuração do dispositivo
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🔧 Dispositivo: {device}")

	# Carrega dados
	print("\n📊 Carregando dados CIFAR-10...")
	train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
	print(f"   • Classes: {len(classes)}")
	print(f"   • Train batches: {len(train_loader)}")
	print(f"   • Val batches: {len(val_loader)}")
	print(f"   • Test batches: {len(test_loader)}")

	# Parâmetros (podem ser ajustados conforme a otimização)
	optimized_params = {
		"num_classes": 10,
		"min_channels": 32,
		"max_channels": 128,
		"dropout_rate": 0.05,
		"num_layers": 4,
		"batch_norm": True,
	}

	print("\n🎯 Parâmetros da ResNet:")
	for key, value in optimized_params.items():
		print(f"   • {key}: {value}")

	params = ResNetParams(**optimized_params)
	model = generate_resnet_architecture(params).to(device)

	# Configurações de treinamento (robustas e configuráveis)
	training_config = {
		"num_epochs": 100,
		"learning_rate": 1e-3,
		"weight_decay": 1e-4,
		"use_mixed_precision": True,
		"use_compile": True,
		"early_stopping_patience": 10,
		"gradient_accumulation_steps": 1,
		"max_grad_norm": 1.0,
		"scheduler_type": "cosine",  # 'cosine' | 'step' | 'plateau'
		"scheduler_kwargs": {"T_max": 100, "eta_min": 1e-6},
		"save_best_model": True,
		"experiment_name": "resnet_full",
	}

	print("\n⚙️  Configuração de treinamento:")
	for key, value in training_config.items():
		print(f"   • {key}: {value}")

	# Otimizador, critério e scheduler
	optimizer = optim.AdamW(
		model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"]
	)
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	scheduler = get_optimized_scheduler(
		optimizer,
		scheduler_type=training_config["scheduler_type"],
		**training_config.get("scheduler_kwargs", {}),
	)

	# Treinamento
	print("\n🚀 Iniciando treinamento...")
	metrics = train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		criterion=criterion,
		optimizer=optimizer,
		num_epochs=training_config["num_epochs"],
		device=device,
		use_mixed_precision=training_config["use_mixed_precision"],
		gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
		max_grad_norm=training_config["max_grad_norm"],
		early_stopping_patience=training_config["early_stopping_patience"],
		scheduler=scheduler,
		compile_model=training_config["use_compile"],
	)

	# Avaliação final
	print("\n🧪 Avaliando no conjunto de teste...")
	final_metrics = evaluate_model(model=model, test_loader=test_loader, criterion=criterion, device=device)

	# Resultado estruturado
	results = {
		"experiment_id": str(uuid.uuid4()),
		"timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
		"model": "ResNet",
		"resnet_params": params.model_dump(),
		"training_config": training_config,
		"history": metrics,
		"test_metrics": final_metrics,
		"classes": classes,
	}

	results_dir = os.path.join("results", "resnet_experiments")
	os.makedirs(results_dir, exist_ok=True)

	results_file = os.path.join(results_dir, f"experiment_{results['experiment_id']}.json")
	with open(results_file, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)
	print(f"💾 Resultados salvos em: {results_file}")

	# Opcional: salvar pesos do melhor modelo (se houver mecanismo de best)
	# torch.save(model.state_dict(), os.path.join(results_dir, f"experiment_{results['experiment_id']}_weights.pt"))

	print("\n🏆 RESULTADOS FINAIS:")
	print(f"   • Top-1 Accuracy: {final_metrics['top1_acc']:.2f}%")
	print(f"   • Top-5 Accuracy: {final_metrics['top5_acc']:.2f}%")
	print(f"   • F1-Score: {final_metrics['f1_macro']:.4f}")
	print(f"   • Loss: {final_metrics['loss']:.4f}")
	print(f"   • Parâmetros: {final_metrics['total_params']:.2e}")
	print(f"   • Tempo de Inferência: {final_metrics['avg_inference_time']:.4f}s")
	print(f"   • Memória: {final_metrics['memory_used_mb']:.2f} MB")
	print(f"   • GFLOPs: {final_metrics['gflops']:.4f}")

	return final_metrics


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Treinamento completo ResNet no CIFAR-10")
	# Hiperparâmetros via CLI (opcional)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--wd", type=float, default=1e-4)
	parser.add_argument("--mixed", action="store_true")
	parser.add_argument("--compile", action="store_true")
	args = parser.parse_args()

	# Ajuste rápido via CLI
	# (Para mudanças mais profundas, edite o dicionário training_config acima)
	final_metrics = train_resnet_full()
