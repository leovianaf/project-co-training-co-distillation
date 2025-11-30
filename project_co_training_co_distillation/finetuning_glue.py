import os
import torch
import argparse
from datasets import load_from_disk
from transformers import (
  AutoTokenizer,
  AutoModelForSequenceClassification,
  DataCollatorWithPadding,
  TrainingArguments,
  Trainer,
)
from .utils_finetune import TASK_CONFIGS, TASK_CONFIGS_PT, get_compute_metrics_fn

# ============================================================
#  Função Principal de Fine-Tuning
# ============================================================

def main():
  # --- Argumentos da Linha de Comando ---
  parser = argparse.ArgumentParser(description="Fine-tune CTCD models on GLUE/ASSIN2 tasks.")
  parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    choices=["teacher", "student"],
    help="Tipo de modelo a ser treinado."
  )
  parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    choices=["stsb", "mrpc", "rte"],
    help="Tarefa para fine-tuning."
  )
  parser.add_argument(
    "--language",
    type=str,
    default="pt",
    choices=["pt", "en"],
    help="Idioma dos dados (pt=ASSIN2, en=GLUE)."
  )
  args = parser.parse_args()

  if args.language == "pt":
    if args.task_name not in TASK_CONFIGS_PT:
      raise ValueError(f"Tarefa {args.task_name} não suportada em Português (use stsb ou rte).")
    TASK_CONFIG = TASK_CONFIGS_PT[args.task_name]
  else:
    if args.task_name not in TASK_CONFIGS:
      raise ValueError(f"Tarefa {args.task_name} não configurada.")
    TASK_CONFIG = TASK_CONFIGS[args.task_name]

  # --- Caminhos Dinâmicos ---
  PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  MODEL_DIR = os.path.join(PROJECT_DIR, "models", f"best_{args.model_type}_model")
  dataset_name = TASK_CONFIG.get("dataset_name", args.task_name)
  DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks", dataset_name)
  OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "finetuned", args.language, args.model_type, args.task_name)

  os.makedirs(OUTPUT_DIR, exist_ok=True)

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Dispositivo: {DEVICE}")
  print(f"[MODELO] {args.model_type} | PATH : {MODEL_DIR}")
  print(f"[TASK-DATASET] {args.task_name} | PATH : {DATA_PATH}")
  print(f"[IDIOMA] {args.language}")

  # --- Carregar e Tokenizar Dataset ---
  if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset {args.task_name} não encontrado em {DATA_PATH}, execute make_dataset.py primeiro.")

  dataset = load_from_disk(DATA_PATH)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

  def preprocess_function(examples):
    return tokenizer(
      examples["sentence1"],
      examples["sentence2"],
      truncation=True,
      max_length=128,
      padding="max_length",
    )

  encoded_ds = dataset.map(preprocess_function, batched=True)
  if args.language == "pt":
    label_col = TASK_CONFIG["label_column"] # ex: 'relatedness_score' ou 'entailment_judgment'

    if args.task_name == "stsb":
      # Para Regressão: Apenas renomeia a coluna de score para 'label'
      print(f"Renomeando coluna '{label_col}' para 'label'...")
      encoded_ds = encoded_ds.rename_column(label_col, "label")

    elif args.task_name == "rte":
      # Para Classificação: Converte String -> Int e renomeia
      print(f"Mapeando labels de '{label_col}' para inteiros...")
      # Mapeamento do ASSIN 2
      label2id = {"NONE": 0, "ENTAILMENT": 1}

      encoded_ds = encoded_ds.map(
          lambda x: {'label': label2id[x[label_col]]}
      )

  # Remove colunas desnecessárias para evitar erro no Trainer
  cols_to_keep = ['input_ids', 'attention_mask', 'token_type_ids', 'label']
  cols_to_remove = [c for c in encoded_ds['train'].column_names if c not in cols_to_keep] # type: ignore
  encoded_ds = encoded_ds.remove_columns(cols_to_remove)

  encoded_ds.set_format("torch")

  # --- Carregar o Modelo que será Fine-tunado ---
  print(f"\nCarregando {args.model_type} para Fine-tuning {args.task_name} ...")
  model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=TASK_CONFIG["num_labels"],
    problem_type=TASK_CONFIG["problem_type"],
  ).to(DEVICE)

  # --- Argumentos do Treino ---
  training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=TASK_CONFIG["metric_name"],
    greater_is_better=TASK_CONFIG["greater_is_better"],
    learning_rate=2e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir=os.path.join(PROJECT_DIR, "reports", "logs"),
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to=["tensorboard"],
    seed=42,
  )

  # --- Configurações do Trainer ---
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["validation"],
    tokenizer=tokenizer, # type: ignore
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=get_compute_metrics_fn(args.task_name),
  )

  # --- Treinar o modelo ---
  print(f"\nIniciando fine-tuning do {args.model_type} na task {args.task_name} ...")
  trainer.train()

  # --- Salvar o modelo fine-tunado ---
  trainer.save_model(OUTPUT_DIR)
  tokenizer.save_pretrained(OUTPUT_DIR)
  print(f"\nModelo fine-tunado salvo em: {OUTPUT_DIR}")
  print(f"Fine-tuning do {args.model_type} concluído com sucesso!")

if __name__ == "__main__":
  main()