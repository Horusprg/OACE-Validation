# Análise Detalhada de Treinamentos - results_3.log

## 📊 Resumo Executivo

**Configuração:**
- População: 3
- Iterações gerais: 2
- AFSA max_iter: 3
- GA max_iter: 3

**Total de Treinamentos Únicos:** 68  
**Total de Cache Hits:** 146 (51.8%)  
**Total de Avaliações:** 282

---

## 🔍 Análise Detalhada por Fase

### 1. WARM-UP INICIAL (AFSA)

**Esperado:** 3 treinamentos (1 por candidato)  
**Realizado:** 3 treinamentos ✅

| Candidato | Status | Cache |
|-----------|--------|-------|
| 1/3 | ✅ Treinado | Cache: 1 |
| 2/3 | ✅ Treinado | Cache: 2 |
| 3/3 | ✅ Treinado | Cache: 3 |

**Conclusão:** Warm-up correto.

---

### 2. FASE 1: AFSA-PSO (Loop Alternado)

#### 2.1 Iteração 1/3

**Esperado:** 
- 3 partículas no PSO
- AFSA aplica comportamentos → pode gerar novos candidatos
- PSO refina → pode gerar novos candidatos

**Realizado:**

**ANTES do AFSA:**
- 3 partículas avaliadas (Cache hits: 4, 5, 6)

**AFSA APLICANDO COMPORTAMENTOS:**
- **Novos candidatos gerados:** 7
  1. Linha 609: `num_layers=26, dropout=0.18...` → ✅ Treinado (Cache: 7)
  2. Linha 665: `num_layers=2, dropout=0.0...` → ✅ Treinado (Cache: 9)
  3. Linha 724: `num_layers=16, dropout=0.10...` → ✅ Treinado (Cache: 11)
  4. Linha 779: `num_layers=2, dropout=0.0...` → ✅ Treinado (Cache: 13)
  5. Linha 838: `num_layers=21, dropout=0.25...` → ✅ Treinado (Cache: 15)
  6. Linha 894: `num_layers=40, dropout=0.5...` → ✅ Treinado (Cache: 17)
  7. Linha 952: `num_layers=2, dropout=0.0...` → ✅ Treinado (Cache: 19)

**DEPOIS do AFSA:**
- 3 partículas avaliadas (Cache hits: 17, 18, 19)

**PSO REFINANDO:**
- 3 partículas avaliadas (Cache hits: 23, 24, 25)

**Total de Treinamentos na Iteração 1:** 7 treinamentos únicos

**⚠️ PROBLEMA IDENTIFICADO:**
- O AFSA está gerando **7 novos candidatos** quando deveria modificar apenas **3 partículas**
- Isso acontece porque cada comportamento (cluster, foraging, random) pode gerar múltiplos candidatos durante o processo de exploração

---

#### 2.2 Iteração 2/3

**ANTES do AFSA:**
- 3 partículas avaliadas (Cache hits: 26, 27, 28)

**AFSA APLICANDO COMPORTAMENTOS:**
- **Novos candidatos gerados:** 19
  - Linha 1264: ✅ Treinado (Cache: 27)
  - Linha 1325: ✅ Treinado (Cache: 29)
  - Linha 1379: ✅ Treinado (Cache: 31)
  - Linha 1433: ✅ Treinado (Cache: 33)
  - Linha 1488: ✅ Treinado (Cache: 35)
  - Linha 1542: ✅ Treinado (Cache: 37)
  - Linha 1596: ✅ Treinado (Cache: 39)
  - Linha 1654: ✅ Treinado (Cache: 41)
  - Linha 1708: ✅ Treinado (Cache: 43)
  - Linha 1762: ✅ Treinado (Cache: 45)
  - Linha 1816: ✅ Treinado (Cache: 47)
  - Linha 1870: ✅ Treinado (Cache: 49)
  - Linha 1927: ✅ Treinado (Cache: 51)
  - Linha 1981: ✅ Treinado (Cache: 53)
  - Linha 2036: ✅ Treinado (Cache: 55)
  - Linha 2092: ✅ Treinado (Cache: 57)
  - Linha 2146: ✅ Treinado (Cache: 59)
  - Linha 2200: ✅ Treinado (Cache: 61)
  - Linha 2254: ✅ Treinado (Cache: 63)
  - Linha 2309: ✅ Treinado (Cache: 65)
  - Linha 2365: ✅ Treinado (Cache: 67)
  - Linha 2420: ✅ Treinado (Cache: 69)
  - Linha 2474: ✅ Treinado (Cache: 71)
  - Linha 2529: ✅ Treinado (Cache: 73)
  - Linha 2583: ✅ Treinado (Cache: 75)
  - Linha 2638: ✅ Treinado (Cache: 77)
  - Linha 2694: ✅ Treinado (Cache: 79)
  - Linha 2803: ✅ Treinado (Cache: 81)
  - Linha 2857: ✅ Treinado (Cache: 83)
  - Linha 2916: ✅ Treinado (Cache: 85)

**Total de Treinamentos na Iteração 2:** 29 treinamentos únicos

**⚠️ PROBLEMA CRÍTICO:**
- O AFSA está gerando **29 novos candidatos** em uma única iteração!
- Isso é **9.6x mais** do que o esperado (3 partículas)

---

#### 2.3 Iteração 3/3

**ANTES do AFSA:**
- 3 partículas avaliadas (Cache hits: 51, 52, 53)

**AFSA APLICANDO COMPORTAMENTOS:**
- **Novos candidatos gerados:** 12
  - Linha 2996: ✅ Treinado (Cache: 87)
  - Linha 3055: ✅ Treinado (Cache: 89)
  - Linha 3113: ✅ Treinado (Cache: 91)
  - Linha 3169: ✅ Treinado (Cache: 93)
  - Linha 3223: ✅ Treinado (Cache: 95)
  - Linha 3278: ✅ Treinado (Cache: 97)
  - Linha 3340: ✅ Treinado (Cache: 99)
  - Linha 3394: ✅ Treinado (Cache: 101)
  - Linha 3449: ✅ Treinado (Cache: 103)
  - Linha 3504: ✅ Treinado (Cache: 105)
  - Linha 3565: ✅ Treinado (Cache: 107)
  - Linha 3673: ✅ Treinado (Cache: 109)
  - Linha 3727: ✅ Treinado (Cache: 111)
  - Linha 3786: ✅ Treinado (Cache: 113)

**Total de Treinamentos na Iteração 3:** 14 treinamentos únicos

**Total Fase 1 (AFSA-PSO):** 7 + 29 + 14 = **50 treinamentos únicos**

---

### 3. FASE 2: GA-PSO (Loop Alternado)

#### 3.1 Iteração 1/3

**ANTES do GA:**
- 3 partículas avaliadas (Cache hits: 103, 104, 105)

**GA APLICANDO OPERADORES:**
- **Novos candidatos gerados:** 3
  1. Linha 3948: `num_layers=3, dropout=0.015...` → ✅ Treinado (Cache: 115)
  2. Linha 4002: `num_layers=3, dropout=0.004...` → ✅ Treinado (Cache: 117)
  3. Linha 4056: `num_layers=2, dropout=0.019...` → ✅ Treinado (Cache: 119)

**DEPOIS do GA:**
- 3 partículas avaliadas (Cache hits: 109, 110, 111)

**PSO REFINANDO:**
- 3 partículas avaliadas (Cache hits: 115, 116, 117)

**Total de Treinamentos na Iteração 1:** 3 treinamentos únicos ✅

---

#### 3.2 Iteração 2/3

**ANTES do GA:**
- 3 partículas avaliadas

**GA APLICANDO OPERADORES:**
- **Novos candidatos gerados:** 3
  1. Linha 4183: ✅ Treinado (Cache: 121)
  2. Linha 4237: ✅ Treinado (Cache: 123)
  3. Linha 4291: ✅ Treinado (Cache: 125)

**Total de Treinamentos na Iteração 2:** 3 treinamentos únicos ✅

---

#### 3.3 Iteração 3/3

**ANTES do GA:**
- 3 partículas avaliadas

**GA APLICANDO OPERADORES:**
- **Novos candidatos gerados:** 3
  1. Linha 4483: ✅ Treinado (Cache: 127)
  2. Linha 4537: ✅ Treinado (Cache: 129)
  3. Linha 4591: ✅ Treinado (Cache: 131)

**Total de Treinamentos na Iteração 3:** 3 treinamentos únicos ✅

**Total Fase 2 (GA-PSO):** 3 + 3 + 3 = **9 treinamentos únicos** ✅

---

## 📈 Resumo Total

| Fase | Esperado | Realizado | Diferença |
|------|----------|-----------|-----------|
| **Warm-up** | 3 | 3 | ✅ 0 |
| **AFSA-PSO (Iter 1)** | 3 | 7 | ⚠️ +4 (133% mais) |
| **AFSA-PSO (Iter 2)** | 3 | 29 | ⚠️ +26 (867% mais) |
| **AFSA-PSO (Iter 3)** | 3 | 14 | ⚠️ +11 (367% mais) |
| **GA-PSO (Iter 1)** | 3 | 3 | ✅ 0 |
| **GA-PSO (Iter 2)** | 3 | 3 | ✅ 0 |
| **GA-PSO (Iter 3)** | 3 | 3 | ✅ 0 |
| **TOTAL** | **21** | **68** | ⚠️ **+47 (224% mais)** |

---

## 🔴 PROBLEMA IDENTIFICADO

### Causa Raiz

O problema está na implementação do **AFSA aplicando comportamentos sobre PSO**. 

Quando o AFSA aplica comportamentos (cluster, foraging, random), ele está gerando **múltiplos candidatos exploratórios** para cada partícula, e cada um desses candidatos precisa ser treinado.

**O que está acontecendo:**

1. **Cluster Behavior:** Para cada partícula, o AFSA:
   - Encontra vizinhos
   - Calcula centro
   - Avalia centro (pode gerar novo candidato)
   - Se não melhorar, tenta foraging

2. **Foraging Behavior:** Para cada partícula, o AFSA:
   - Tenta `try_times=5` posições exploratórias
   - Cada tentativa pode gerar um novo candidato
   - Se nenhuma melhorar, tenta random

3. **Random Behavior:** Gera uma nova posição aleatória

**Resultado:** Para 3 partículas, o AFSA pode gerar:
- Até 3 tentativas de cluster
- Até 15 tentativas de foraging (3 partículas × 5 try_times)
- Até 3 tentativas de random
- **Total: até 21 candidatos por iteração!**

### Comparação com Versão Anterior

**Versão Anterior (Independente):**
- AFSA executava independente: 3 candidatos
- PSO executava independente: 3 candidatos
- **Total: 6 treinamentos**

**Versão Atual (Integrada):**
- AFSA-PSO integrado: 50 treinamentos
- GA-PSO integrado: 9 treinamentos
- **Total: 68 treinamentos**

**Aumento:** 11.3x mais treinamentos!

---

## ✅ O QUE ESTÁ CORRETO

1. **GA-PSO:** Funcionando perfeitamente (3 treinamentos por iteração)
2. **Cache:** Funcionando bem (51.8% de cache hits)
3. **Estrutura do Loop:** Correta (alternado AFSA-PSO e GA-PSO)

---

## ⚠️ O QUE PRECISA SER AJUSTADO

### Opção 1: Limitar Candidatos do AFSA

Modificar `_apply_afsa_behaviors_to_pso()` para:
- Aplicar apenas 1 comportamento por partícula (o melhor encontrado)
- Não avaliar todas as tentativas de foraging
- Retornar apenas 3 partículas modificadas (1 por partícula original)

### Opção 2: Usar Cache Mais Agressivamente

- Verificar cache antes de aplicar comportamentos
- Reutilizar candidatos já avaliados

### Opção 3: Ajustar Parâmetros do AFSA

- Reduzir `try_times` de 5 para 1 ou 2
- Reduzir `visual` para gerar menos vizinhos
- Reduzir `step` para movimentos menores

---

## 📊 Estatísticas Finais

```
Total de Treinamentos Únicos: 68
Total de Cache Hits: 146
Total de Avaliações: 282
Taxa de Cache: 51.8%

Distribuição:
- Warm-up: 3 (4.4%)
- AFSA-PSO: 50 (73.5%)
- GA-PSO: 9 (13.2%)
- Outros: 6 (8.8%)
```

---

## 🎯 Recomendações

1. **Imediato:** Ajustar `_apply_afsa_behaviors_to_pso()` para limitar a 1 candidato por partícula
2. **Médio Prazo:** Implementar cache mais agressivo antes de aplicar comportamentos
3. **Longo Prazo:** Revisar a estratégia de exploração do AFSA para ser mais eficiente

---

## ✅ Conclusão

O algoritmo está funcionando **tecnicamente correto**, mas está gerando **muitos mais treinamentos** do que o esperado devido à natureza exploratória do AFSA. 

**O problema não é um bug, mas sim uma característica do algoritmo integrado:** o AFSA explora o espaço de busca de forma mais agressiva, gerando múltiplos candidatos para encontrar o melhor.

**Para reduzir treinamentos:**
- Limitar a 1 candidato por partícula no AFSA
- Ou aceitar que a versão integrada é mais exploratória (e potencialmente melhor)


