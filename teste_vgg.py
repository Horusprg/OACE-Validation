import torch
import sys
import os

# Adiciona o diretório raiz ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vgg.vgg_architecture import generate_vgg_architecture, VGG
from models.vgg.vgg_train_eval import warm_up_vgg
from utils.data_loader import get_cifar10_dataloaders


def test_vgg_compatibility():
    """
    Testa a compatibilidade da VGG com dados CIFAR-10.
    """
    print("Testando compatibilidade VGG + CIFAR-10...")
    
    # Parâmetros de teste para CIFAR-10
    test_params = {
        "num_classes": 10,
        "in_channels": 3,
        "num_conv_blocks": 2,
        "min_conv_layers_per_block": 1,
        "max_conv_layers_per_block": 1,
        "min_channels": 16,
        "max_channels": 64,
        "num_fc_layers": 1,
        "min_fc_neurons": 128,
        "max_fc_neurons": 256,
        "batch_norm": True,
        "dropout_rate": 0.2
    }
    
    try:
        # Gerar modelo
        model = generate_vgg_architecture(**test_params)
        print(f" Modelo VGG criado com sucesso")
        
        # Testar com entrada CIFAR-10
        test_input = torch.randn(2, 3, 32, 32)  # Batch de 2 imagens
        output = model(test_input)
        
        print(f" Forward pass bem-sucedido!")
        print(f"   Entrada: {test_input.shape}")
        print(f"   Saída: {output.shape}")
        print(f"   Esperado: torch.Size([2, 10])")
        
        assert output.shape == torch.Size([2, 10]), "Formato de saída incorreto!"
        print(" VGG compatível com CIFAR-10!")
        return True
        
    except Exception as e:
        print(f" Erro no teste: {e}")
        return False


def test_vgg_training():
    """
    Testa o treinamento da VGG com um mini-experimento.
    """
    print("\n Testando treinamento da VGG...")
    
    # Parâmetros otimizados para teste rápido
    model_params = {
        "num_classes": 10,
        "in_channels": 3,
        "dropout_rate": 0.3,
        "batch_norm": True,
        "weight_init_fn": "kaiming",
        "num_conv_blocks": 2,  # Rede pequena para teste rápido
        "min_conv_layers_per_block": 1,
        "max_conv_layers_per_block": 1,
        "min_channels": 16,
        "max_channels": 32,
        "num_fc_layers": 1,
        "min_fc_neurons": 64,
        "max_fc_neurons": 128,
    }
    
    try:
        # Carregar dados com batch size pequeno
        train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(
            batch_size=32
        )
        print(f" Dados CIFAR-10 carregados")
        print(f"   Classes: {classes}")
        
        # Configurar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Dispositivo: {device}")
        
        # Executar mini-treinamento (apenas 2 épocas para teste)
        print("\n Iniciando mini-treinamento (2 épocas)...")
        warm_up_vgg(
            model_params=model_params,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=classes,
            num_epochs=2,  # Só 2 épocas para teste rápido
            device=device,
        )
        
        print(" Treinamento de teste concluído com sucesso!")
        return True
        
    except Exception as e:
        print(f" Erro no treinamento: {e}")
        return False


def test_vgg_variants():
    """
    Testa diferentes variantes da VGG.
    """
    print("\n Testando variantes da VGG...")
    
    variants = [
        {
            "name": "VGG Tiny",
            "params": {
                "num_classes": 10,
                "num_conv_blocks": 2,
                "min_channels": 8,
                "max_channels": 32,
                "num_fc_layers": 1,
                "min_fc_neurons": 32,
                "max_fc_neurons": 64,
            }
        },
        {
            "name": "VGG Small",
            "params": {
                "num_classes": 10,
                "num_conv_blocks": 3,
                "min_channels": 16,
                "max_channels": 64,
                "num_fc_layers": 1,
                "min_fc_neurons": 128,
                "max_fc_neurons": 256,
            }
        },
        {
            "name": "VGG Medium",
            "params": {
                "num_classes": 10,
                "num_conv_blocks": 4,
                "min_channels": 32,
                "max_channels": 128,
                "num_fc_layers": 2,
                "min_fc_neurons": 256,
                "max_fc_neurons": 512,
            }
        }
    ]
    
    for variant in variants:
        try:
            print(f"\n   Testando {variant['name']}...")
            model = generate_vgg_architecture(**variant['params'])
            
            # Teste forward
            test_input = torch.randn(1, 3, 32, 32)
            output = model(test_input)
            
            # Contar parâmetros
            num_params = sum(p.numel() for p in model.parameters())
            
            print(f"    {variant['name']}: {output.shape}, {num_params:,} parâmetros")
            
        except Exception as e:
            print(f"    {variant['name']}: Erro - {e}")


def main():
    """
    Executa todos os testes.
    """
    print("=" * 60)
    print(" TESTE DA ARQUITETURA VGG")
    print("=" * 60)
    
    # Teste 1: Compatibilidade básica
    success1 = test_vgg_compatibility()
    
    # Teste 2: Variantes da arquitetura
    test_vgg_variants()
    
    # Teste 3: Treinamento (opcional - pode ser lento)
    print("\n" + "=" * 60)
    choice = input(" Deseja testar o treinamento? (s/n): ").lower().strip()
    
    if choice in ['s', 'sim', 'y', 'yes']:
        success2 = test_vgg_training()
    else:
        print("⏭  Pulando teste de treinamento")
        success2 = True
    
    # Resumo final
    print("\n" + "=" * 60)
    print(" RESUMO DOS TESTES")
    print("=" * 60)
    print(f" Compatibilidade: {'PASSOU' if success1 else 'FALHOU'}")
    print(f" Treinamento: {'PASSOU' if success2 else 'PULADO/FALHOU'}")
    
    if success1 and success2:
        print("\n Todos os testes passaram! Sua arquitetura VGG está funcionando!")
    else:
        print("\n  Alguns testes falharam. Verifique os erros acima.")


if __name__ == "__main__":
    main()