import os
import uuid
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.Inception.inception_architecture import InceptionV3, generate_inception_architecture
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model


def warm_up_inception(
    model: InceptionV3,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classes: list,
    num_epochs: int,
    device: torch.device,
) -> None:
    """
    Script warm_up para treinar e avaliar uma EfficientNet gerada aleatoriamente no CIFAR-10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Configuração randômica
    model = generate_inception_architecture(
        num_classes=10,
        dropout_rate=0.7,
        aux_logits=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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
        "model": "InceptionV3",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classes": classes,
    }

    results_dir = "results/inception_experiments"
    os.makedirs(results_dir, exist_ok=True)

    # Salva resultados em JSON
    results_file = os.path.join(
        results_dir, f'experiment_{results["experiment_id"]}.json'
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em: {results_file}")
 

def specialized_training_inception():
    pass