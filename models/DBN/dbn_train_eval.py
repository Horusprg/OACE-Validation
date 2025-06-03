import os
import uuid
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.DBN.dbn_architecture import DBN, generate_dbn_architecture
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model

def warm_up_dbn(model: DBN, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, classes: list, num_epochs: int, device: torch.device) -> None:
    """
    Script warm_up para treinar e avaliar uma DBN no CIFAR-10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configuração randômica
    model = generate_dbn_architecture(input_dim=3072,
                                    output_dim=10,
                                    num_rbm_layers=3,
                                    min_rbm_neurons=128,
                                    max_rbm_neurons=512,
                                    num_classifier_hidden_layers=3,
                                    min_classifier_neurons=128,
                                    max_classifier_neurons=512,
                                    rbm_activation_function_choice='sigmoid',
                                    classifier_activation_function_choice='relu',
                                    dropout_rate=0.2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_metrics = train_model(model=model, 
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=num_epochs, 
                                device=device)

    test_metrics = evaluate_model(model=model, 
                                  test_loader=test_loader,
                                  criterion=criterion,
                                  device=device)
    
    results = {
        'experiment_id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'DBN',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes
    }
    
    results_dir = 'results/dbn_experiments'
    os.makedirs(results_dir, exist_ok=True)

    # Salva resultados em JSON
    results_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em: {results_file}")

def specialized_training_dbn(model: DBN, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, classes: list, num_epochs: int, device: torch.device) -> None:
    pass