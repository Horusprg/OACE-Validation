import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import uuid
from datetime import datetime
from models.vgg.vgg_architecture import generate_vgg_architecture, VGG
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model


def warm_up_vgg(
    model_params: dict,  # Parâmetros para gerar a VGG
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classes: list,
    num_epochs: int,
    device: torch.device,
) -> None:
    """
    Script warm_up para treinar e avaliar uma VGG genérica no CIFAR-10.

    Args:
        model_params: Dicionário de parâmetros para a função generate_vgg_architecture.
        train_loader: DataLoader para o conjunto de treinamento.
        val_loader: DataLoader para o conjunto de validação.
        test_loader: DataLoader para o conjunto de teste.
        classes: Lista de nomes das classes.
        num_epochs: Número de épocas para treinamento.
        device: Dispositivo para treinamento (CPU ou CUDA).
    """
    print("\n--- Iniciando warm-up da VGG ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gera uma nova instância da VGG com base nos parâmetros
    try:
        model = generate_vgg_architecture(**model_params).to(device)
    except NameError:
        print(
            "Erro: A função 'generate_vgg_architecture' não está definida ou acessível."
        )
        print("Por favor, garanta que ela está importada ou definida no escopo.")
        return  # Impede a continuação se a função não estiver disponível

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Treinamento do modelo
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Avaliação do modelo
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    # Coleta e salva resultados
    results = {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_architecture": "VGG",
        "model_parameters": model_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classes": classes,
    }

    results_dir = "results/vgg_experiments"
    os.makedirs(results_dir, exist_ok=True)

    # Salva resultados em JSON
    results_file = os.path.join(
        results_dir, f'experiment_{results["experiment_id"]}.json'
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em: {results_file}")
    print("--- Warm-up da VGG Concluído ---")


def specialized_training_vgg(
    model: VGG, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> None:
    """
    Função placeholder para treinamento especializado de uma VGG.
    Adicione sua lógica de treinamento especializado aqui, se necessário.
    """
    print("\n--- Iniciando treinamento especializado da VGG (placeholder) ---")
    print("--- Treinamento especializado da VGG Concluído (placeholder) ---")
