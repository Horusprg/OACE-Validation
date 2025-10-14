from models.ResNet.resnet_train_eval import warm_up_resnet
from models.ResNet.resnet_architecture import ResNet
from models.MLP.mlp_train_eval import warm_up_mlp
from models.MLP.mlp_architecture import MLP
from models.DBN.dbn_architecture import DBN
from models.DBN.dbn_train_eval import warm_up_dbn
from utils.data_loader import get_cifar10_dataloaders
from models.Inception.inception_train_eval import warm_up_inception
from models.Inception.inception_architecture import InceptionV3
from models.EfficientNet.efficientnet_train_eval import warm_up_efficientnet
from models.EfficientNet.efficientnet_architecture import EfficientNet
import torch
from utils.training_evaluation_pipeline import train_and_evaluate_for_oace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device principal: {device}")

train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(n_valid=0.2, batch_size=64, num_workers=0)


warm_up_mlp(
        model=MLP	,
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        classes=classes,
        num_epochs=3, 
        device=device)
"""
config = {
        "model_type": "DBN",
        "input_dim": 3072,
            "output_dim": 10,
            "num_rbm_layers": 2,
            "min_rbm_neurons": 64,
            "max_rbm_neurons": 256,
            "num_classifier_hidden_layers": 2,
            "min_classifier_neurons": 64,
            "max_classifier_neurons": 256,
            "rbm_activation_function_choice": "sigmoid",
            "classifier_activation_function_choice": "relu",
            "dropout_rate": 0.8
}

metrics = train_and_evaluate_for_oace(config, train_loader, val_loader, device, epochs=1)

print(metrics)"""






