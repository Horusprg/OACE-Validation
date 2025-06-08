import torch
from utils.data_loader import get_cifar10_dataloaders
import matplotlib.pyplot as plt
import numpy as np

def test_cifar10_data():
    # Carregar os dados
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=4)
    
    # Pegar um batch de treino
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Mostrar informações básicas
    print(f"Número de classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Shape das imagens: {images.shape}")
    print(f"Labels do batch: {[classes[label] for label in labels]}")
    
    # Mostrar algumas imagens
    plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        # Desnormalizar
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cifar10_sample.png')
    plt.close()
    
    # Verificar tamanhos dos datasets
    print(f"\nTamanho do dataset de treino: {len(train_loader.dataset)}")
    print(f"Tamanho do dataset de validação: {len(val_loader.dataset)}")
    print(f"Tamanho do dataset de teste: {len(test_loader.dataset)}")

if __name__ == "__main__":
    test_cifar10_data() 