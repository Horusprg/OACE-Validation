import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader



def get_cifar10_dataloaders(n_valid=0.2, batch_size=64, num_workers=0):
    """10 Classes"""
    
    transform_train = transforms.Compose([transforms.ToTensor(),
                                    #transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),                                    
                                    transforms.RandomCrop(32, padding=4),   
                                    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  
    ])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

    n_train = len(train_data)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    split = int(np.floor(n_valid * n_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Prepare data loaders (combine dataset and sampler)
    trainLoader = torch.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = train_sampler,
                                            num_workers = num_workers)

    validLoader = torch.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = valid_sampler,
                                            num_workers = num_workers)

    testLoader = torch.utils.data.DataLoader(test_data,
                                            batch_size = batch_size,
                                            num_workers = num_workers)

    classes = train_data.classes
    return trainLoader, validLoader, testLoader, classes

# Definir a classe do dataset FORA da função
class WildShapesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_wildshapes_dataloaders(batch_size=64, num_workers=0):
    """
    Carrega o dataset WildShapes e retorna os DataLoaders.
    
    Args:
        batch_size (int): Tamanho do batch. Padrão: 64
        num_workers (int): Número de workers para carregamento de dados. 
                          Padrão: 0 (recomendado para Windows).
                          Use valores maiores (ex: 4, 8) em Linux/Mac para melhor performance.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
    """
    
    # Mesmas transformações
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Carregar e dividir
    ds = load_dataset("Horusprg/WildShapes")
    
    split1 = ds['train'].train_test_split(test_size=0.3, seed=42, stratify_by_column='label')
    split2 = split1['test'].train_test_split(test_size=1/3, seed=42, stratify_by_column='label')
    
    final_ds = DatasetDict({
        'train': split1['train'],
        'validation': split2['train'],
        'test': split2['test']
    })
    
    # Criar datasets
    train_dataset = WildShapesDataset(final_ds['train'], transform_train)
    val_dataset = WildShapesDataset(final_ds['validation'], transform_test)
    test_dataset = WildShapesDataset(final_ds['test'], transform_test)
    
    # DataLoaders com num_workers configurável
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False if num_workers == 0 else True  # pin_memory só funciona com num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    classes = [f'class_{i}' for i in range(9)]
    
    print(f"WildShapes Dataset")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val: {len(val_dataset):,}")
    print(f"  Test: {len(test_dataset):,}")
    print(f"  Batch: {batch_size}")
    print(f"  Num Workers: {num_workers}")
    
    return train_loader, val_loader, test_loader, classes