import os
import json
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
from .utils_finetune import get_compute_metrics_fn

# ============================================================
#  Função Principal de Avaliação
# ============================================================

def main():
  # --- Argumentos da Linha de Comando ---
  parser = argparse.ArgumentParser(description="Avaliar modelos fine-tunados nas tarefas GLUE.")
  parser.add_argument("--model_type", type=str, required=True, choices=["teacher", "student"])
  parser.add_argument("--task_name", type=str, required=True, choices=["stsb", "mrpc", "rte"])
  parser.add_argument(
      "--split",
      type=str,
      default="validation",
      choices=["validation", "test"],
      help="Split do dataset para avaliar."
  )
  args = parser.parse_args()

  # --- Caminhos Dinâmicos ---
  PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  MODEL_DIR = os.path.join(PROJECT_DIR, "models", "finetuned", args.model_type, args.task_name)
  DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks", args.task_name)

  print(f"Avaliando {args.model_type} / {args.task_name} no split '{args.split}'")
  print(f"Carregando modelo de: {MODEL_DIR}")

  # --- Carregar Modelo, Tokenizer e Dataset ---
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
  dataset = load_from_disk(DATA_PATH)

  def preprocess_function(examples):
    return tokenizer(
      examples["sentence1"],
      examples["sentence2"],
      truncation=True,
      max_length=128,
      padding="max_length",
    )

  encoded_ds = dataset.map(preprocess_function, batched=True)
  encoded_ds = encoded_ds.remove_columns(["sentence1", "sentence2", "idx"])
  encoded_ds.set_format("torch")

  # --- Configurar Trainer (mínimo para avaliação) ---
  training_args = TrainingArguments(
      output_dir=os.path.join(MODEL_DIR, "eval_temp"), # Diretório temporário
      per_device_eval_batch_size=32,
      report_to=[],
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_ds[args.split],
    tokenizer=tokenizer, # type: ignore
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=get_compute_metrics_fn(args.task_name),
  )

  # --- Avaliar ---
  print("Iniciando avaliação...")
  metrics = trainer.evaluate(encoded_ds[args.split]) # type: ignore

  # SALVAR MÉTRICAS COMO JSON para o script de visualização
  save_path = os.path.join(MODEL_DIR, "all_results.json")
  with open(save_path, "w") as f:
      json.dump(metrics, f, indent=2)
  print(f"[OK] Métricas salvas em: {save_path}")

  print(f"\n--- Métricas Finais ({args.model_type} / {args.task_name} / {args.split}) ---")
  print(metrics)

if __name__ == "__main__":
  main()
