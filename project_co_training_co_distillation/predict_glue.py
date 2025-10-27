import os
import torch
import argparse
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from .utils_finetune import get_compute_metrics_fn

# ============================================================
#  Predição no conjunto de teste (sem labels)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Gerar previsões no conjunto de teste GLUE (sem labels).")
    parser.add_argument("--model_type", type=str, required=True, choices=["teacher", "student"])
    parser.add_argument("--task_name", type=str, required=True, choices=["stsb", "mrpc", "rte"])
    args = parser.parse_args()

    # --- Caminhos ---
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(PROJECT_DIR, "models", "finetuned", args.model_type, args.task_name)
    DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks", args.task_name)
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "reports", "predictions")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[INFO] Gerando predições - Modelo: {args.model_type.upper()} | Task: {args.task_name.upper()}")

    # --- Carregar modelo e tokenizer ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    # --- Carregar dataset ---
    dataset = load_from_disk(DATA_PATH)
    if "test" not in dataset:
        raise ValueError(f"Dataset {args.task_name} não possui split 'test'.")

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    encoded_ds = dataset.map(preprocess_function, batched=True)
    encoded_ds = encoded_ds.remove_columns(
        [col for col in ["sentence1", "sentence2", "idx"] if col in encoded_ds["test"].column_names]
    )
    if "label" in encoded_ds["test"].column_names:
        encoded_ds["test"] = encoded_ds["test"].remove_columns(["label"])

    encoded_ds.set_format("torch")

    # --- Configuração mínima de inferência ---
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "pred_temp"),
        per_device_eval_batch_size=32,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # --- Predição ---
    print("[INFO] Iniciando predições no conjunto de teste...")
    predictions = trainer.predict(encoded_ds["test"])

    logits = predictions.predictions
    if args.task_name == "stsb":
        # STSB é uma tarefa de regressão
        preds = logits.squeeze()
    else:
        # MRPC / RTE são classificações
        preds = logits.argmax(axis=-1)

    # --- Salvar resultados ---
    df_out = pd.DataFrame({
        "id": range(len(preds)),
        "prediction": preds.tolist()
    })

    out_path = os.path.join(OUTPUT_DIR, f"{args.model_type}_{args.task_name}_test_predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Predições salvas em: {out_path}")

if __name__ == "__main__":
    main()
