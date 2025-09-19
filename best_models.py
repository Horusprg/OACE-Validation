import os
import json

def encontrar_melhores_arquiteturas(diretorio, metrica, top_n=3):
    """
    Analisa arquivos JSON de experimentos em um diretório, classifica-os
    por uma métrica específica e retorna os N melhores.

    Args:
        diretorio (str): O caminho para a pasta contendo os arquivos JSON.
        metrica (str): A chave da métrica a ser usada para classificação (ex: 'precision_macro').
        top_n (int): O número de melhores resultados a serem retornados.

    Returns:
        list: Uma lista de dicionários contendo as informações dos top N experimentos.
    """
    
    # 1. Lista para armazenar os resultados de cada experimento válido
    todos_os_experimentos = []

    print(f"🔍 Analisando arquivos no diretório: '{diretorio}'...")

    # 2. Verifica se o diretório existe
    if not os.path.isdir(diretorio):
        print(f"❌ Erro: O diretório '{diretorio}' não foi encontrado.")
        return []

    # 3. Itera sobre cada arquivo na pasta especificada
    for filename in os.listdir(diretorio):
        # Processa apenas arquivos com extensão .json (ou sem extensão, se for o caso)
        # Vamos assumir que os arquivos JSON podem não ter a extensão .json
        filepath = os.path.join(diretorio, filename)
        
        # Garante que é um arquivo e não um subdiretório
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 4. Extrai as informações necessárias do JSON
                    experiment_id = data.get("experiment_id")
                    test_metrics = data.get("test_metrics", {})
                    valor_metrica = test_metrics.get(metrica)

                    # Verifica se a métrica alvo foi encontrada
                    if valor_metrica is not None:
                        todos_os_experimentos.append({
                            "id": experiment_id,
                            "arquivo": filename,
                            metrica: float(valor_metrica)
                        })
                    else:
                        print(f"⚠️ Aviso: Métrica '{metrica}' não encontrada no arquivo '{filename}'. Pulando.")

            except json.JSONDecodeError:
                print(f"⚠️ Aviso: Erro ao decodificar JSON no arquivo '{filename}'. Pulando.")
            except KeyError as e:
                print(f"⚠️ Aviso: Chave {e} não encontrada no arquivo '{filename}'. Pulando.")
            except Exception as e:
                print(f"⚠️ Aviso: Ocorreu um erro inesperado ao processar '{filename}': {e}")

    # 5. Verifica se algum experimento foi carregado
    if not todos_os_experimentos:
        print("❌ Nenhum experimento válido foi encontrado para análise.")
        return []
        
    print(f"\n✅ Análise concluída. {len(todos_os_experimentos)} experimentos processados com sucesso.")

    # 6. Ordena a lista de experimentos em ordem decrescente com base na métrica
    experimentos_ordenados = sorted(todos_os_experimentos, key=lambda x: x[metrica], reverse=True)
    
    # 7. Retorna os 'top_n' melhores resultados
    return experimentos_ordenados[:top_n]

# --- Execução do Script ---
if __name__ == "__main__":
    # Caminho para a pasta de resultados dos seus experimentos
    # Ajuste este caminho conforme a estrutura do seu projeto
    diretorio_experimentos = os.path.join("results", "cnn_experiments")
    
    # Métrica que queremos maximizar
    metrica_alvo = "precision_macro"
    
    # Número de melhores modelos a selecionar
    numero_de_candidatos = 3

    # Chama a função principal
    melhores_candidatos = encontrar_melhores_arquiteturas(
        diretorio=diretorio_experimentos,
        metrica=metrica_alvo,
        top_n=numero_de_candidatos
    )

    # 8. Exibe os resultados
    if melhores_candidatos:
        print(f"\n🏆 Top {len(melhores_candidatos)} Melhores Arquiteturas (baseado em '{metrica_alvo}'):\n")
        for i, candidato in enumerate(melhores_candidatos):
            print(f"--- Posição #{i+1} ---")
            print(f"  ID do Experimento: {candidato['id']}")
            print(f"  Arquivo JSON: {candidato['arquivo']}")
            print(f"  Valor de {metrica_alvo}: {candidato[metrica_alvo]:.6f}")
            print("-" * (18 + len(str(i+1))))