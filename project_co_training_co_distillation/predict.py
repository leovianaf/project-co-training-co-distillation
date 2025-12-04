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
from .utils_finetune import TASK_CONFIGS, TASK_CONFIGS_PT


# ============================================================
#  Predição no conjunto de teste (PT + EN) — Student & Teacher
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Gerar previsões no conjunto de teste (GLUE/ASSIN2).")

    # AGORA SUPORTA TEACHER
    parser.add_argument("--model_type", type=str, required=True, choices=["teacher", "student"])

    parser.add_argument("--task_name", type=str, required=True, choices=["stsb", "mrpc", "rte"])
    parser.add_argument("--language", type=str, required=True, choices=["pt", "en"])
    args = parser.parse_args()

    # ------------------------------------------------------------
    #  Seleção da configuração da task
    # ------------------------------------------------------------
    if args.language == "pt":
        if args.task_name not in TASK_CONFIGS_PT:
            raise ValueError(
                f"A task '{args.task_name}' não está disponível em PT (use stsb ou rte)."
            )
        TASK_CONFIG = TASK_CONFIGS_PT[args.task_name]

    else:  # inglês
        if args.task_name not in TASK_CONFIGS:
            raise ValueError(
                f"A task '{args.task_name}' não existe no GLUE/EN."
            )
        TASK_CONFIG = TASK_CONFIGS[args.task_name]

    dataset_name = TASK_CONFIG.get("dataset_name", args.task_name)

    # ------------------------------------------------------------
    #  Caminhos
    # ------------------------------------------------------------
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

    OUTPUT_DIR = os.path.join(PROJECT_DIR, "reports", "predictions")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n===========================================")
    print(f"[PREDICT] Modelo     : {args.model_type.upper()}")
    print(f"[PREDICT] Task       : {args.task_name.upper()}")
    print(f"[PREDICT] Idioma     : {args.language.upper()}")
    print(f"[PREDICT] Dataset    : {dataset_name}")
    print(f"[PREDICT] Model Path : {MODEL_DIR}")
    print("===========================================\n")

    # ------------------------------------------------------------
    #  Carregar modelo + tokenizer
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    # ------------------------------------------------------------
    #  Carregar dataset
    # ------------------------------------------------------------
    dataset = load_from_disk(DATA_PATH)

    if "test" not in dataset:
        raise ValueError(f"O dataset '{dataset_name}' não tem split 'test'.")

    # ------------------------------------------------------------
    #  Tokenização
    # ------------------------------------------------------------
    def preprocess_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    encoded_ds = dataset.map(preprocess_function, batched=True)

    # remover colunas desnecessárias
    cols_to_drop = []
    for col in ["sentence1", "sentence2", "idx", "label"]:
        if col in encoded_ds["test"].column_names:
            cols_to_drop.append(col)

    encoded_ds = encoded_ds.remove_columns(cols_to_drop)
    encoded_ds.set_format("torch")

    # ------------------------------------------------------------
    #  Configuração para inferência
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    #  Predição
    # ------------------------------------------------------------
    print("[INFO] Rodando predições...")
    predictions = trainer.predict(encoded_ds["test"])

    logits = predictions.predictions

    if args.task_name == "stsb":
        preds = logits.squeeze()  # Regressão
    else:
        preds = logits.argmax(axis=-1)  # Classificação

    # ------------------------------------------------------------
    #  Salvar arquivo final
    # ------------------------------------------------------------
    df_out = pd.DataFrame({
        "id": list(range(len(preds))),
        "prediction": preds.tolist()
    })

    out_file = f"{args.model_type}_{args.task_name}_{args.language}_test_predictions.csv"
    out_path = os.path.join(OUTPUT_DIR, out_file)

    df_out.to_csv(out_path, index=False)

    print(f"[OK] Predições salvas em: {out_path}")


if __name__ == "__main__":
    main()
