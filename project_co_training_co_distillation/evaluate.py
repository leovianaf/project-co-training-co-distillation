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
from .utils_finetune import TASK_CONFIGS, TASK_CONFIGS_PT, get_compute_metrics_fn

# ============================================================
#  Avaliação (GLUE EN + ASSIN2 PT)
# ============================================================

def main():
    # --- Argumentos da Linha de Comando ---
    parser = argparse.ArgumentParser(description="Avaliar modelos fine-tunados (GLUE / ASSIN2).")
    parser.add_argument("--model_type", type=str, required=True, choices=["teacher", "student"])
    parser.add_argument("--task_name", type=str, required=True, choices=["stsb", "mrpc", "rte"])
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["pt", "en"],
        help="Idioma dos dados (pt=ASSIN2, en=GLUE)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Split do dataset para avaliar."
    )
    args = parser.parse_args()

    # -------- Selecionar config da task --------
    if args.language == "pt":
        if args.task_name not in TASK_CONFIGS_PT:
            raise ValueError(f"Tarefa {args.task_name} não suportada em PT.")
        TASK_CONFIG = TASK_CONFIGS_PT[args.task_name]
    else:
        if args.task_name not in TASK_CONFIGS:
            raise ValueError(f"Tarefa {args.task_name} não suportada em EN.")
        TASK_CONFIG = TASK_CONFIGS[args.task_name]

    dataset_name = TASK_CONFIG.get("dataset_name", args.task_name)

    # -------- Caminhos --------
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    MODEL_DIR = os.path.join(
        PROJECT_DIR,
        "models",
        "finetuned",
        args.language,
        args.model_type,
        args.task_name,
    )

    DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks", dataset_name)

    print(f"\n[Avaliando] {args.model_type} / {args.task_name} / {args.language} no split '{args.split}'")
    print(f"[Modelo]    {MODEL_DIR}")
    print(f"[Dados]     {DATA_PATH}\n")

    # -------- Carregar modelo + tokenizer + dataset --------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    dataset = load_from_disk(DATA_PATH)

    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' não existe no dataset '{dataset_name}'.")

    # -------- Tokenização --------
    def preprocess_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    encoded_ds = dataset.map(preprocess_function, batched=True)

    # -------- Ajustar labels para PT (ASSIN2) --------
    if args.language == "pt":
        label_col = TASK_CONFIG["label_column"]  # 'relatedness_score' ou 'entailment_judgment'

        if args.task_name == "stsb":
            # Regressão: renomeia para 'label' e converte para float
            def map_stsb(example):
                # relatedness_score é normalmente int (0–5); convertemos para float
                return {"label": float(example[label_col])}

            encoded_ds = encoded_ds.map(map_stsb)
            # remover coluna original
            encoded_ds = encoded_ds.remove_columns([label_col])

        elif args.task_name == "rte":
            # Classificação binária: mapear strings para inteiros e renomear para 'label'
            label2id = {
                "NONE": 0,
                "ENTAILMENT": 1,
                0: 0,
                1: 1,
            }

            def map_rte(example):
                return {"label": label2id[example[label_col]]}

            encoded_ds = encoded_ds.map(map_rte)
            encoded_ds = encoded_ds.remove_columns([label_col])

    # -------- Remover colunas que não entram no modelo --------
    cols_to_drop = []
    for col in ["sentence1", "sentence2", "idx"]:
        if col in encoded_ds[args.split].column_names:
            cols_to_drop.append(col)

    if cols_to_drop:
        encoded_ds = encoded_ds.remove_columns(cols_to_drop)

    encoded_ds.set_format("torch")

    # -------- Configurar Trainer --------
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "eval_temp"),
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

    # -------- Avaliar --------
    print("[INFO] Iniciando avaliação...\n")
    metrics = trainer.evaluate(encoded_ds[args.split]) # type: ignore

    save_path = os.path.join(MODEL_DIR, "all_results.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Métricas salvas em: {save_path}")
    print(f"\n--- Métricas Finais ({args.model_type} / {args.task_name} / {args.language} / {args.split}) ---")
    print(metrics)


if __name__ == "__main__":
    main()
