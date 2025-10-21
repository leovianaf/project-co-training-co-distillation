# project-co-training-co-distillation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A replication study of the paper 'Co-training and Co-distillation for Quality Improvement and Compression of Language Models'.

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md          <- Este arquivo.
├── data
│   ├── external       <- Dados de fontes externas.
│   ├── interim        <- Dados intermediários.
│   ├── processed      <- Datasets finais processados (ex: amostras locais, dataset tokenizado).
│   ├── raw            <- Dados brutos originais.
│   └── tasks          <- Dados para rodar as tasks GLUE de fine-tuning.
│
├── docs               <- Documentação completa do projeto.
│
├── models             <- Modelos treinados, checkpoints e saídas.
│   ├── best_student_model <- Melhor modelo Student após pré-treinamento CTCD.
│   ├── best_teacher_model <- Melhor modelo Teacher após pré-treinamento CTCD.
│   ├── ctcd-sample-output <- Checkpoints do pré-treinamento CTCD.
│   └── finetuned          <- Modelos após fine-tuning nas tarefas GLUE.
│       ├── student
│       └── teacher
│
├── notebooks          <- Jupyter notebooks para exploração e prototipagem.
│   ├── data-download.ipynb         <- Notebook para download dos datasets completos.
│   └── make_sample_datasets.ipynb  <- Notebook (Colab) para criar amostras dos datasets.
│
├── pyproject.toml     <- Configuração do projeto (ruff, build, etc.).
│
├── references         <- Materiais de referência
│   └── 2023.findings-emnlp.500.pdf. <- Paper original.
│
├── reports            <- Relatórios, figuras e logs de treinamento.
│   └── figures        <- Gráficos e tabelas de resultados.
│   └── logs           <- Logs do TensorBoard.
│
├── requirements.txt   <- Dependências do Python.
│
└── project_co_training_co_distillation   <- Código-fonte do projeto.
    │
    ├── __init__.py
    │
    ├── ctcd                 <- Módulo específico do pré-treinamento CTCD.
    │   ├── __init__.py
    │   └── train_model.py   <- Script para pré-treinamento CTCD (Teacher + Student).
    │
    ├── make_dataset.py      <- Script para baixar e preparar todos os datasets.
    ├── finetuning_glue.py   <- Script reutilizável para fine-tuning em tarefas GLUE.
    ├── evaluate_glue.py     <- Script reutilizável para avaliar modelos em tarefas GLUE.
    └── utils_finetune.py    <- Funções utilitárias para fine-tuning e avaliação.
```

--------

## Usage Pipeline

Este projeto replica o framework CTCD. O pipeline principal consiste em:

1.  **Preparação dos Dados:** Baixar e processar o corpus de pré-treinamento e os datasets de avaliação GLUE.
2.  **Pré-treinamento CTCD:** Treinar os modelos Teacher (6-camadas BERT) e Student (4-camadas BERT) simultaneamente usando a loss CTCD.
3.  **Fine-tuning:** Ajustar os modelos pré-treinados (Teacher ou Student) em tarefas específicas do GLUE (STS-B, MRPC, RTE).
4.  **Avaliação:** Medir o desempenho dos modelos fine-tunados nos conjuntos de teste das tarefas GLUE.

### Comandos Principais

Execute os seguintes comandos a partir da **raiz do projeto**:

1.  **Preparar todos os datasets (pré-treino e GLUE):**
    ```bash
    python -m project_co_training_co_distillation.make_dataset
    ```
    *(Este script baixa os datasets GLUE e processa os dados de pré-treino locais salvos em `data/processed`)*

2.  **Executar o pré-treinamento CTCD (com a amostra de dados):**
    ```bash
    python -m project_co_training_co_distillation.ctcd.train_model
    ```
    *(Isso treinará Teacher e Student juntos e salvará os melhores modelos em `models/best_teacher_model` e `models/best_student_model`)*

3.  **Executar o Fine-tuning:**
    Use o script `finetuning_glue.py` com os argumentos `--model_type` e `--task_name`.

    **Argumentos `--model_type` disponíveis:**

    | Argumento | Descrição                      |
    | :-------- | :----------------------------- |
    | `teacher` | Usa o modelo Teacher (6-layer) |
    | `student` | Usa o modelo Student (4-layer) |

    **Argumentos `--task_name` disponíveis:**

    | Argumento | Descrição da Tarefa GLUE                                               |
    | :-------- | :--------------------------------------------------------------------- |
    | `stsb`    | STS-B: Similaridade Semântica Textual (regressão)                      |
    | `mrpc`    | MRPC: Detecção de Paráfrases (classificação binária)                   |
    | `rte`     | RTE: Reconhecimento de Implicação Textual (classificação binária)      |

    **Exemplo (Student no STS-B):**
    ```bash
    python -m project_co_training_co_distillation.finetuning_glue --model_type student --task_name stsb
    ```
    *(Carrega `models/best_student_model`, faz fine-tuning na tarefa STS-B e salva o resultado em `models/finetuned/student/stsb`)*

4.  **Executar a Avaliação:**
    Use o script `evaluate_glue.py` com os argumentos `--model_type`, `--task_name` e opcionalmente `--split`.

    **Argumento `--split` disponíveis:**

    | Argumento    | Descrição                                 |
    | :----------- | :---------------------------------------- |
    | `validation` | Usa o conjunto de validação (padrão)      |
    | `test`       | Usa o conjunto de teste                   |

    **Exemplo (Student no STS-B, split de teste):**
    ```bash
    python -m project_co_training_co_distillation.evaluate_glue --model_type student --task_name stsb --split test
    ```
    *(Carrega o modelo de `models/finetuned/student/stsb` e avalia no conjunto de teste do STS-B)*
