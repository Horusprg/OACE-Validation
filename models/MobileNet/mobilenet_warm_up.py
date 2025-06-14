import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.MobileNet.mobilenet_architecture import generate_mobilenet_architecture
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model
from models.MobileNet.mobilenet_architecture import MobilenetParams

def warm_up_mobilenet(
    train_loader,
    val_loader,
    test_loader,
    classes,
    num_epochs=3,
    device=None,
    params=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = generate_mobilenet_architecture(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    results = {
        'experiment_id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'MobileNet',
        'mobilenet_params': params.dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes
    }

    results_dir = 'results/mobilenet_experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Resultados salvos em: {results_file}")

    weights_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}_weights.pt')
    torch.save(model.state_dict(), weights_file)
    print(f"Pesos do modelo salvos em: {weights_file}")

    return test_metrics
