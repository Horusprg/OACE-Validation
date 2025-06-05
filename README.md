# Otimização de Arquiteturas de Redes Neurais (NAS) para CIFAR-10 com AFSA-GA-PSO e OACE

Este projeto explora uma abordagem inovadora para a Otimização de Arquiteturas de Redes Neurais (NAS - Neural Architecture Search), utilizando uma combinação de algoritmos de otimização heurística (Artificial Fish Swarm Algorithm - AFSA, Genetic Algorithm - GA e Particle Swarm Optimization - PSO) e um método de avaliação de desempenho multicritério chamado Optimized Assertiveness-Cost Evaluation (OACE). O objetivo principal é encontrar arquiteturas de redes neurais otimizadas para o dataset CIFAR-10, que apresentem um equilíbrio superior entre alta assertividade (precisão) e baixo custo computacional.

## 🌟 Visão Geral do Projeto

A tarefa de encontrar a arquitetura de rede neural ideal para uma dada aplicação é complexa e exige um balanço cuidadoso entre desempenho e eficiência. Modelos de Machine Learning, especialmente os de Deep Learning, frequentemente atingem alta assertividade às custas de um elevado custo computacional, o que pode ser um problema para ambientes com recursos limitados. Este projeto visa superar esse desafio através de uma metodologia de NAS que:

1.  **Otimiza a Estrutura da Rede Neural:** Utiliza uma abordagem híbrida de otimização meta-heurística (AFSA-GA-PSO) para explorar eficientemente o vasto espaço de busca de arquiteturas.
2.  **Avalia Modelos com OACE:** Emprega o método OACE para avaliar holisticamente as arquiteturas candidatas, combinando métricas de assertividade (precisão, acurácia, recall) e custo computacional (parâmetros totais, tempo de inferência, tamanho do modelo) em um único score balanceado.
3.  **Define Pesos de Métricas com AHP:** Aplica o Analytic Hierarchy Process (AHP) para determinar os pesos ideais das métricas de assertividade e custo dentro da função OACE, garantindo uma avaliação objetiva e ponderada.
4.  **Foco no CIFAR-10:** Aplica e valida a metodologia no dataset CIFAR-10, um benchmark popular para classificação de imagens.

## 🚀 Metodologia

O projeto segue um pipeline de otimização de NAS que integra os seguintes componentes:

### 1. Representação da Arquitetura Neural

As arquiteturas de redes neurais são representadas de forma a permitir a manipulação pelos algoritmos de otimização. Parâmetros estruturais como o número de camadas ocultas e o número de neurônios/filtros por camada são codificados como "partículas" (no contexto do PSO) ou "indivíduos" (no contexto do GA).

### 2. Algoritmo Híbrido de Otimização (AFSA-GA-PSO)

Nosso método de otimização combina a força de três algoritmos heurísticos:

* **Particle Swarm Optimization (PSO):** Utilizado como a base do algoritmo de busca, com partículas explorando o espaço de soluções com base em suas melhores experiências pessoais (`pbest`) e a melhor experiência global (`gbest`) do enxame[cite: 325, 329, 330].
* **Artificial Fish Swarm Algorithm (AFSA):** Aplicado para otimizar a inicialização das partículas do PSO, utilizando comportamentos de aglomeração, forrageamento e busca aleatória para encontrar soluções iniciais mais promissoras[cite: 217, 345, 361]. Isso ajuda a evitar que o PSO caia em ótimos locais prematuramente[cite: 363].
* **Genetic Algorithm (GA):** Introduzido após a fase AFSA-PSO para refinar a busca global. Operadores genéticos como crossover e mutação são aplicados para aumentar a diversidade do enxame de partículas e superar problemas de convergência prematura do PSO puro[cite: 218, 371]. As probabilidades de crossover e mutação são ajustadas dinamicamente durante o processo[cite: 380, 385].

Este algoritmo híbrido busca um equilíbrio entre a exploração de novas áreas do espaço de busca e a exploração de soluções promissoras, visando encontrar a melhor arquitetura de rede neural.

### 3. Avaliação de Desempenho com OACE

O **Optimized Assertiveness-Cost Evaluation (OACE)** é a pedra angular da nossa função de aptidão. Ele permite uma avaliação holística das arquiteturas candidatas, combinando métricas de:

* **Assertividade:** Precisão, Acurácia e Recall, que medem a capacidade do modelo de reconhecer padrões nos dados e minimizar erros de classificação.
* **Custo Computacional:** Parâmetros Totais do Modelo (MTP), Tempo por Inferência (TPI) e Tamanho do Modelo (MS), que avaliam a eficiência e a leveza do modelo, crucial para ambientes com recursos limitados.

A função OACE agrega essas métricas em um único score $S_{\phi}$, usando a fórmula $S_{\phi}(m)=\lambda \cdot A(m) + (1-\lambda) \cdot C(m)$, onde $\lambda$ é um parâmetro de controle que balanceia a importância entre assertividade ($A(m)$) e custo ($C(m)$). Um $\lambda$ maior que 0.5 é preferível para maximizar a assertividade enquanto minimiza o custo.

### 4. Pesos de Métricas com AHP

Para garantir uma ponderação objetiva das métricas de assertividade e custo dentro do OACE, o **Analytic Hierarchy Process (AHP)** é empregado[cite: 87]. O AHP permite a derivação de pesos de prioridade para cada métrica com base em comparações paritárias, transformando um problema complexo em uma hierarquia de critérios[cite: 88, 90].

### 5. Treinamento e Avaliação das Arquiteturas Candidatas

Cada arquitetura proposta pelo algoritmo de otimização é treinada no dataset CIFAR-10. Para acelerar o processo de busca, um número reduzido de épocas ou um subconjunto dos dados pode ser utilizado para a avaliação inicial. Após o treinamento, as métricas de assertividade e custo são coletadas, e o score OACE é calculado para determinar a "aptidão" daquela arquitetura.

## 📊 Estrutura do Projeto

```
.
├── config.py                   # Configurações de parâmetros dos algoritmos e do OACE
├── search.py                   # Script principal para iniciar o NAS
├── train_eval.py               # Script para treinamento e avaliação do melhor modelo encontrado
├── optimize.py                 # Otimização dos hiperparâmetros do melhor modelo encontrado
├── models/
│   ├── CapsNet/                # Definição da estrutura e codificação da arquitetura
│   ├── CNN/                    # Definição da estrutura e codificação da arquitetura
│   ├── DBN/                    # Definição da estrutura e codificação da arquitetura
│   ├── DenseNet/               # Definição da estrutura e codificação da arquitetura
│   ├── EfficientNet/           # Definição da estrutura e codificação da arquitetura
│   ├── Inception/              # Definição da estrutura e codificação da arquitetura
│   ├── MLP/                    # Definição da estrutura e codificação da arquitetura
│   ├── MobileNet/              # Definição da estrutura e codificação da arquitetura
│   ├── ResNet/                 # Definição da estrutura e codificação da arquitetura
│   ├── VGG/                    # Definição da estrutura e codificação da arquitetura
│   ├── ViT/                    # Definição da estrutura e codificação da arquitetura
├── optimizers/
│   ├── pso.py                  # Implementação do Particle Swarm Optimization
│   ├── afsa.py                 # Implementação do Artificial Fish Swarm Algorithm
│   ├── ga.py                   # Implementação do Genetic Algorithm
│   └── afsa_ga_pso.py          # Implementação do algoritmo híbrido AFSA-GA-PSO
├── results/                    # Pasta para armazenar os resultados dos experimentos
│   └── (experimento_X)/        # Subpastas para cada execução
│       ├── best_architecture.json
│       ├── optimization_log.csv
│       └── ...
└── utils/
    ├── data_loader.py          # Funções para carregar e pré-processar o CIFAR-10
    ├── fitness_function.py     # Função de aptidão (OACE)
    ├── ahp_weights.py          # Implementação do AHP para pesos das métricas
    ├── evaluate_utils.py       # Pipeline de coleta de métricas
    ├── training_utils.py       # Pipeline de treinamento
    ├── best_model_selection.py # Lógica para seleção da melhor arquitetura final
    ├── analyze_results.py      # Scripts para visualização e análise de resultados
    └── (outros utilitários)
```

## 🛠️ Como Executar

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Configurar Ambiente (Consulte `requirements.txt`):**
    ```bash
    pip install -r requirements.txt
    ```
    Certifique-se de ter o ambiente CUDA configurado corretamente se for usar GPU.

3.  **Executar a Otimização NAS:**
    ```bash
    python main.py
    ```
    Você pode ajustar os parâmetros em `config.py` antes de executar.

4.  **Analisar os Resultados:**
    Os resultados dos experimentos serão salvos na pasta `results/`.
    ```bash
    python utils/analyze_results.py
    ```

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues, enviar pull requests ou sugerir melhorias.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.

---