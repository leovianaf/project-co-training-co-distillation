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
│   └── tasks          <- Dados para rodar as tasks ASSIN 2 ou GLUE de fine-tuning.
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
    ├── make_dataset.py      <- Script para baixar e preparar os datasets em inglês.
    ├── make_dataset_pt.py   <- Script para baixar e preparar os datasets em português.
    ├── finetuning_glue.py   <- Script reutilizável para fine-tuning em tarefas GLUE.
    ├── evaluate_glue.py     <- Script reutilizável para avaliar modelos em tarefas GLUE.
    └── utils_finetune.py    <- Funções utilitárias para fine-tuning e avaliação.
```

--------

## Usage Pipeline

Este projeto replica o framework CTCD e foi adaptado para funcionar tanto com o pipeline original (Inglês) quanto para Português (PT-BR).

1.  **Preparação dos Dados:** Baixar e processar o corpus de pré-treinamento e os datasets de avaliação GLUE.
2.  **Pré-treinamento CTCD:** Treinar os modelos Teacher (6-camadas BERT) e Student (4-camadas BERT) simultaneamente usando a loss CTCD.
3.  **Fine-tuning:** Ajustar os modelos pré-treinados (Teacher ou Student) em tarefas específicas do GLUE (STS-B, MRPC, RTE).
4.  **Avaliação:** Medir o desempenho dos modelos fine-tunados nos conjuntos de teste das tarefas GLUE.

### Comandos Principais

Execute os seguintes comandos a partir da **raiz do projeto**:

1.  **Preparar todos os datasets (pré-treino e GLUE):**
    * **Para Português:**
    Baixa o dataset ASSIN 2 e processa a amostra do corpus Aroeira.
    ```bash
    python -m project_co_training_co_distillation.make_dataset_pt
    ```

    * **Para Inglês:**
    Baixa o GLUE e processa amostras do BookCorpus/Wikipedia.
    ```bash
    python -m project_co_training_co_distillation.make_dataset
    ```
    *(Este script baixa os datasets ASSIN 2 ou GLUE e processa os dados de pré-treino locais salvos em `data/processed`, é necessário baixar o Corpus desejado)*

2.  **Executar o pré-treinamento CTCD (com a amostra de dados):**
    O script de treinamento agora aceita argumentos para definir o dataset e o tokenizer.

    **Argumentos Disponíveis:**

    | Argumento          | Descrição                                                       | Exemplo PT-BR                                | Exemplo EN                                |
    | :----------------- | :-------------------------------------------------------------- | :------------------------------------------- | :---------------------------------------- |
    | `--dataset_path`   | Caminho para a pasta do dataset tokenizado (gerado no passo 1). | `data/processed/tokenized_aroeira_subset_1k` | `data/processed/tokenized_sample_dataset` |
    | `--tokenizer_name` | Nome do modelo no Hugging Face para o tokenizador.              | `neuralmind/bert-base-portuguese-cased`      | `bert-base-uncased`                       |

    **Comando para executar (Exemplo em Português para amostra de testes (1000 dados)):**
    ```bash
    python -m project_co_training_co_distillation.ctcd.train_model \
    --dataset_path "data/processed/tokenized_aroeira_subset_1k" \
    --tokenizer_name "neuralmind/bert-base-portuguese-cased"
    ```
    **Comando padrão (Execução em Português por default):**
    ```bash
    python -m project_co_training_co_distillation.ctcd.train_model
    ```
    *(Isso treinará Teacher e Student juntos e salvará os melhores modelos em `models/best_teacher_model` e `models/best_student_model`)*

3.  **Executar o Fine-tuning:**
    Use o script `finetuning.py` com os argumentos `--model_type`, `--task_name` e `--language`.

    **Argumentos `--model_type` disponíveis:**

    | Argumento | Descrição                      |
    | :-------- | :----------------------------- |
    | `teacher` | Usa o modelo Teacher (6-layer) |
    | `student` | Usa o modelo Student (4-layer) |

    **Argumentos `--task_name` disponíveis:**

    | Argumento | Dataset (PT-BR)       | Dataset (EN) | Descrição da Tarefa GLUE                                      |
    | :-------- | :-------------------- | :----------- | :------------------------------------------------------------ |
    | `stsb`    | ASSIN 2 (Similarity)  | STS-B        | Similaridade Semântica Textual (regressão)                    |
    | `rte`     | ASSIN 2 (Entailment)  | RTE          | Reconhecimento de Implicação Textual (classificação binária)  |
    | `mrpc`    | (Não suportado em PT) | MRPC         | Detecção de Paráfrases (classificação binária)                |

    **Argumentos `--language` disponíveis:**

    | Argumento | Descrição                            |
    | :-------- | :----------------------------------- |
    | `pt`      | Usa o dataset em português (ASSIN 2) |
    | `en`      | Usa o dataset em inglês (GLUE)       |

    **Exemplo (Student no ASSIN 2 - Similaridade):**
    ```bash
    python -m project_co_training_co_distillation.finetuning --model_type student --task_name stsb --language pt
    ```
    *(Carrega `models/best_student_model`, faz fine-tuning no ASSIN 2 e salva em `models/finetuned/pt/student/stsb`)*

4.  **Executar a predição no Conjunto de Teste:**
    Use o script `predict.py` com os argumentos `--model_type`, `--task_name` e `--language`.

    Gera um arquivo CSV com as predições do modelo fine-tunado no conjunto de teste. Útil para submissão em leaderboards ou análise qualitativa dos erros.

    **Argumentos Disponíveis: Os mesmos do Fine-tuning (--model_type, --task_name, --language).**

    **Exemplo (Student no STS-B, split de teste):**
    ```bash
    python -m project_co_training_co_distillation.predict --model_type student --task_name stsb --language pt
    ```
    *(O arquivo CSV será salvo na pasta `reports/predictions/`)*

5.  **Executar a Avaliação:**
    Use o script `evaluate.py` com os argumentos `--model_type`, `--task_name` e opcionalmente `--split`.

    **Argumento `--split` disponíveis:**

    | Argumento    | Descrição                                 |
    | :----------- | :---------------------------------------- |
    | `validation` | Usa o conjunto de validação (padrão)      |
    | `test`       | Usa o conjunto de teste                   |

    **Exemplo (Student no STS-B, split de teste):**
    ```bash
    python -m project_co_training_co_distillation.evaluate_glue --model_type student --task_name stsb --language pt --split test
    ```
    *(Carrega o modelo de `models/finetuned/pt/student/stsb` e avalia no conjunto de teste do STS-B (ASSIN2))*
