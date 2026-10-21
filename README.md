# ANN 2026/1 — Redes Neurais Artificiais

Repositório com os trabalhos práticos da disciplina de **Redes Neurais Artificiais (ANN)**, semestre **2026/1**.
Todo o conteúdo está em **Jupyter Notebooks** (Python), com código, gráficos e discussão dos resultados em português.

---

## 📂 Estrutura do repositório

```
ANN-2026-1/
├── T1/                     # Trabalho 1
│   ├── T1_pt1.ipynb        # Parte 1 — Descida de gradiente (1D e 2D)
│   └── T1_pt2.ipynb        # Parte 2 — Classificação de falhas no reator CSTR (Softmax Regression)
├── requirements.txt        # Dependências Python
├── .gitignore
└── README.md
```

Novos trabalhos devem seguir o mesmo padrão: uma pasta por trabalho (`T2/`, `T3/`, ...).

---

## 📘 Trabalho 1

### Parte 1 — Descida de gradiente (`T1/T1_pt1.ipynb`)

Implementação, do zero, do algoritmo de **descida de gradiente**, que é a base do treinamento de redes neurais.

| Questão | O que é feito |
|---|---|
| **Q1** | Um passo de descida de gradiente em $f(x) = e^{-x}\,x(x^2 - x - 1)$ a partir de $x_0 = 1$, aproximando a derivada por **diferenças finitas** ($h = 0{,}01$, $\alpha = 0{,}1$). |
| **Q2** | O mesmo passo, agora com a **derivada analítica** obtida via `sympy.diff` + `lambdify`, e comparação com a aproximação numérica. |
| **Q3** | Descida de gradiente completa a partir de $x_0 = 3$, com critérios de parada $\lvert f'(x)\rvert < g_{\min} = 0{,}1$ e $k_{\max} = 100$. Plota a trajetória $x_k \to f(x_k)$ sobre a curva. |
| **Q4** | Busca de **dois mínimos locais** de uma função bidimensional (variante da *six-hump camel*), usando (a) gradiente por diferenças finitas, (b) gradiente explícito e (c) visualização da trajetória em curvas de nível 2D e superfície 3D. |

**Resultados principais**

- Q1 → $x_1 \approx 0{,}96266$ (gradiente aproximado $\approx 0{,}37337$)
- Q2 → $x_1 \approx 0{,}96321$ (gradiente exato $= 1/e \approx 0{,}36788$)
- Q3 → mínimo em $x \approx 0{,}78494$, $f(x) \approx -0{,}41849$, após 49 iterações
- Q4 → mínimos distintos dependendo do ponto inicial, mostrando que em funções não convexas o resultado depende da inicialização

### Parte 2 — Detecção de falhas em um reator químico (`T1/T1_pt2.ipynb`)

Aplicação de um **classificador linear treinado por descida de gradiente** a um problema real de engenharia: diagnosticar falhas em um **CSTR** (*Continuous Stirred-Tank Reactor*, reator contínuo de tanque agitado).

Os dados são gerados pelo simulador [CSTRSIM](https://github.com/twrauber/CSTRSIM), que modela o reator em malha fechada (controladores, sensores e limites de segurança).

**Pipeline do notebook**

1. **Simulação** — 5 cenários de 500 min cada; nos cenários com falha, ela é ativada aos 200 min:

   | Classe | Falha |
   |---|---|
   | `normal` | Operação normal |
   | `2` | Bloqueio na saída do reator |
   | `3` | Bloqueio na jaqueta de resfriamento |
   | `12` | Vazão anormal de alimentação |
   | `17` | Pressão anormal na saída da jaqueta |

2. **Consolidação** — junção dos CSVs em uma base única de **2505 amostras × 18 atributos** (14 variáveis medidas + 4 restrições físicas).
3. **Análise exploratória** — distribuição das classes (base desbalanceada: `normal` ≈ 52%), checagem de valores ausentes e estatísticas descritivas.
4. **Pré-processamento** — codificação dos rótulos, divisão estratificada **60% treino / 20% validação / 20% teste** e padronização com `StandardScaler` (ajustado só no treino, evitando *data leakage*).
5. **Modelo** — **Regressão Softmax** implementada manualmente com NumPy: entropia cruzada + regularização L2, otimizada por SGD com mini-batches (η = 0,1, 200 épocas, batch = 256, λ = 10⁻⁴).
6. **Avaliação** — curvas de perda/acurácia, relatório de classificação, matrizes de confusão (treino/validação/teste) e métricas ponderadas.

**Resultados no conjunto de teste**

| Métrica (ponderada) | Valor |
|---|---|
| Acurácia | 0,872 |
| Precisão | 0,778 |
| Recall | 0,872 |
| F1-score | 0,819 |

As falhas 2, 3 e 12 são reconhecidas quase perfeitamente. Já a **falha 17 não é detectada** e acaba confundida com a classe `normal`: com as variáveis disponíveis, ela não é linearmente separável. Isso motiva o uso de modelos não lineares (redes neurais) nos próximos trabalhos.

---

## 🚀 Como executar

### Opção 1 — Google Colab (recomendado para a Parte 2)

A Parte 2 foi escrita para o **Google Colab**: ela clona o CSTRSIM, instala o `fortranformat` e muda o diretório para `/content/CSTRSIM/python`.

1. Abra o [Google Colab](https://colab.research.google.com/) → *Arquivo* → *Abrir notebook* → aba *GitHub*.
2. Cole a URL deste repositório e escolha o notebook.
3. Execute as células em ordem (*Ambiente de execução* → *Executar tudo*).

### Opção 2 — Ambiente local

Requisitos: **Python 3.10+** (desenvolvido com 3.12).

```bash
git clone <url-deste-repositorio>
cd ANN-2026-1

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

> **Atenção (Parte 2 local):** a célula `os.chdir('/content/CSTRSIM/python')` usa o caminho do Colab.
> Para rodar localmente, clone o simulador e ajuste o caminho para onde ele estiver na sua máquina, por exemplo:
>
> ```python
> os.chdir('CSTRSIM/python')
> ```
>
> A pasta `CSTRSIM/` já está no `.gitignore`, então ela não será versionada por engano.

---

## 🧰 Tecnologias

- **NumPy**: álgebra linear e implementação dos algoritmos do zero
- **SymPy**: derivação simbólica (gradiente analítico)
- **Matplotlib / Seaborn**: gráficos 2D, 3D, curvas de treino e matrizes de confusão
- **Pandas**: manipulação dos dados simulados
- **scikit-learn**: apenas para divisão dos dados, padronização e métricas (o modelo é implementado manualmente)
- **CSTRSIM**: simulador do reator usado para gerar a base de dados

---

## 📚 Conceitos abordados

- Descida de gradiente e critérios de parada
- Derivada numérica (diferenças finitas) × derivada analítica
- Mínimos locais e dependência da inicialização em funções não convexas
- Classificação multiclasse com Softmax e entropia cruzada
- Regularização L2 e SGD com mini-batches
- Divisão treino/validação/teste, *data leakage* e bases desbalanceadas
- Métricas de classificação: acurácia, precisão, recall, F1 e matriz de confusão
