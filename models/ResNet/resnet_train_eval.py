import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.ResNet.resnet_architecture import ResNet, generate_resnet_architecture
from utils.data_loader import get_cifar10_dataloaders
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model
import uuid


def train_and_evaluate_resnet(
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: torch.device = None
) -> None:
    """
    Função principal para treinar e avaliar uma ResNet gerada aleatoriamente no CIFAR-10.
    
    Args:
        num_epochs (int): Número de épocas para treinamento
        batch_size (int): Tamanho do batch para treinamento
        learning_rate (float): Taxa de aprendizado para o otimizador
        device (torch.device): Dispositivo para treinamento (CPU/GPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gerar arquitetura ResNet aleatória
    model = generate_resnet_architecture().to(device)
    
    # Configurar otimizador e função de perda
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Carregar dados do CIFAR-10
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(
        batch_size=batch_size
    )
    
    # Treinar o modelo
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Avaliar o modelo
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Preparar e salvar resultados
    results = {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model": "ResNet",
        "hyperparameters": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classes": classes,
    }

    # Criar diretório para resultados se não existir
    results_dir = "results/resnet_experiments"
    os.makedirs(results_dir, exist_ok=True)

    # Salvar resultados em JSON
    results_file = os.path.join(
        results_dir, f'experiment_{results["experiment_id"]}.json'
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em: {results_file}")
    return results


if __name__ == "__main__":
    # Exemplo de uso
    train_and_evaluate_resnet(
        num_epochs=10,
        batch_size=128,
        learning_rate=0.001
    )


def specialized_training_resnet(
    model: ResNet, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> None:
    pass
