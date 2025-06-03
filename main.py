from models.ResNet.resnet_train_eval import warm_up_resnet
from models.ResNet.resnet_architecture import ResNet
from models.MLP.mlp_train_eval import warm_up_mlp
from models.MLP.mlp_architecture import MLP
from models.DBN.dbn_architecture import DBN
from models.DBN.dbn_train_eval import warm_up_dbn
from utils.data_loader import get_cifar10_dataloaders
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(n_valid=0.2, batch_size=64, num_workers=0)

warm_up_dbn(model = DBN, 
        train_loader = train_loader, 
        val_loader = val_loader, 
        test_loader = test_loader,
        classes = classes,
        num_epochs = 3, 
        device = torch.device)