import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate


# ============================================================
#  Configurações principais
# ============================================================

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "best_teacher_model")  # modelo salvo após CTCD
DATA_PATH = os.path.join(PROJECT_DIR, "data", "tasks", "stsb")         # dataset local
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "finetuned", "teacher", "stsb")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")
print(f"Carregando modelo CTCD pré-treinado de: {MODEL_DIR}")
print(f"Carregando dataset local de: {DATA_PATH}")


# ============================================================
#  1. Carregar dataset STS-B (local)
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset STS-B não encontrado em {DATA_PATH}\n"
        f"Execute antes o make_dataset.py para baixá-lo."
    )

dataset = load_from_disk(DATA_PATH)
print(f"Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}, Test: {len(dataset['test'])}")


# ============================================================
#  2. Tokenização
# ============================================================

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
encoded_ds = encoded_ds.remove_columns(["sentence1", "sentence2", "idx"])
encoded_ds.set_format("torch")

print("\n✅ Exemplo tokenizado:")
print(encoded_ds["train"][0])


# ============================================================
#  3. Modelo com cabeça de regressão
# ============================================================

print("\nCarregando modelo Teacher para Fine-tuning STS-B ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=1,  # regressão contínua
    problem_type="regression",
).to(DEVICE)


# ============================================================
#  4. Métrica: Spearman correlation (GLUE)
# ============================================================

metric = evaluate.load("glue", "stsb")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    return metric.compute(predictions=predictions, references=labels)


# ============================================================
#  5. Data collator
# ============================================================

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
#  6. Argumentos de treino
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="spearmanr",
    greater_is_better=True,
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
    report_to=[],
    seed=42,
)


# ============================================================
#  7. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ============================================================
#  8. Fine-tuning
# ============================================================

print("\nIniciando fine-tuning do Teacher na task STS-B ...")
trainer.train()

metrics = trainer.evaluate(encoded_ds["validation"])
print("\nMétricas finais (dev set):")
print(metrics)


# ============================================================
#  9. Salvamento do modelo final
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModelo fine-tunado salvo em: {OUTPUT_DIR}")
print("Fine-tuning do Teacher concluído com sucesso!")