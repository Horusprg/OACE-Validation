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


def warm_up_resnet(
    model: ResNet,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classes: list,
    num_epochs: int,
    device: torch.device,
) -> None:
    """
    Script warm_up para treinar e avaliar uma ResNet gerada aleatoriamente no CIFAR-10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = generate_resnet_architecture().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # configuração randômica
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model, test_loader=test_loader, criterion=criterion, device=device
    )

    results = {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model": "ResNet",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classes": classes,
    }

    results_dir = "results/resnet_experiments"
    os.makedirs(results_dir, exist_ok=True)

    # Salva resultados em JSON
    results_file = os.path.join(
        results_dir, f'experiment_{results["experiment_id"]}.json'
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em: {results_file}")


def specialized_training_resnet(
    model: ResNet, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> None:
    pass
