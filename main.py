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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device principal: {device}")

train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(n_valid=0.2, batch_size=64, num_workers=0)

warm_up_inception(
        model=InceptionV3,
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        classes=classes,
        num_epochs=3, 
        device=device)