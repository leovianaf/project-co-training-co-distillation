import os
import math
import inspect
from itertools import chain
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    BertConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ============================================================
#  Custom Trainer: Co-Training + Co-Distillation
# ============================================================
class CTCDTrainer(Trainer):
    """
    Trainer customizado para Co-Training + Co-Distillation.
    - self.model         -> Teacher
    - self.student_model -> Student
    """

    def __init__(self, student_model: torch.nn.Module, *args, temperature=2.0, lambda_soft=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = student_model
        self.temperature = float(temperature)
        self.lambda_soft = float(lambda_soft)
        self.student_model.to(self.args.device)

    # Otimizador conjunto (teacher + student)
    def create_optimizer(self):
        if self.optimizer is None:
            opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            params = chain(self.model.parameters(), self.student_model.parameters()) # type: ignore
            self.optimizer = opt_cls(params, **opt_kwargs)
        return self.optimizer

    # Scheduler padrão
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None): # type: ignore
        return super().create_scheduler(num_training_steps, optimizer or self.optimizer)

    # Loss conjunta (co-training + co-distillation)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_model = model
        student_model = self.student_model

        teacher_model.train()
        student_model.train()

        outputs_teacher = teacher_model(**inputs)
        outputs_student = student_model(**inputs)

        hard_loss_teacher = outputs_teacher.loss
        hard_loss_student = outputs_student.loss

        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("Esperava 'labels' em inputs para MLM.")

        mask = labels.ne(-100)
        if not mask.any():
            soft_loss_student = torch.zeros((), device=labels.device)
            soft_loss_teacher = torch.zeros((), device=labels.device)
        else:
            t_logits = outputs_teacher.logits[mask]
            s_logits = outputs_student.logits[mask]
            T = self.temperature

            # KL(Student || Teacher): treina TEACHER
            soft_loss_teacher = F.kl_div(
                input=F.log_softmax(t_logits / T, dim=-1),
                target=F.softmax(s_logits.detach() / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)

            # KL(Teacher || Student): treina STUDENT
            soft_loss_student = F.kl_div(
                input=F.log_softmax(s_logits / T, dim=-1),
                target=F.softmax(t_logits.detach() / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)

        loss_teacher = hard_loss_teacher + self.lambda_soft * soft_loss_teacher
        loss_student = hard_loss_student + self.lambda_soft * soft_loss_student
        total_loss = loss_teacher + loss_student

        return (total_loss, outputs_teacher) if return_outputs else total_loss


# ============================================================
#  Função de Avaliação separada (Teacher vs Student)
# ============================================================
def evaluate_model(model, tokenizer, dataset, batch_size=8):
    """
    Calcula loss média e perplexity em um dataset, usando o mesmo collator do treino.
    """
    model.eval()
    device = next(model.parameters()).device

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * len(batch["input_ids"])
            total_count += len(batch["input_ids"])

    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, perplexity


# ============================================================
#  Pre-training com Co-Training + Co-Distillation
# ============================================================
def main():
    print(f"PyTorch CUDA disponível? {torch.cuda.is_available()}")

    project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    tokenized_dataset_path = os.path.join(project_dir, "data", "processed", "tokenized_sample_dataset")
    out_dir = os.path.join(project_dir, "models", "ctcd-sample-output")

    # Dataset
    print(f"Carregando dataset de: {tokenized_dataset_path}")
    ds = load_from_disk(tokenized_dataset_path)
    split = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Modelos Teacher e Student
    print("Inicializando modelos do zero (BERT-MLM) ...")
    teacher_config = BertConfig(
        vocab_size=30522,
        num_hidden_layers=6,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )
    student_config = BertConfig(
        vocab_size=30522,
        num_hidden_layers=4,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )

    teacher_model = AutoModelForMaskedLM.from_config(teacher_config)
    student_model = AutoModelForMaskedLM.from_config(student_config)

    # Tokenizer e collator
    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Args de treino
    args_kwargs = {
        "output_dir": out_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 10,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "logging_steps": 10,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "fp16": torch.cuda.is_available(),
        "report_to": [],
        "seed": 42,
        "logging_strategy": "steps",
    }

    sig = inspect.signature(TrainingArguments)
    if "evaluation_strategy" in sig.parameters:
        args_kwargs["evaluation_strategy"] = "epoch"
        args_kwargs["save_strategy"] = "epoch"
    else:
        args_kwargs["eval_strategy"] = "epoch"
        args_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(**args_kwargs)

    # Trainer customizado
    trainer = CTCDTrainer(
        model=teacher_model,
        student_model=student_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        temperature=2.0,
        lambda_soft=1.0,
    )

    # Treinamento
    print("Iniciando co-training + co-distillation ...")
    trainer.train()

    # Salvando os melhores modelos
    best_teacher_dir = os.path.join(project_dir, "models", "best_teacher_model")
    best_student_dir = os.path.join(project_dir, "models", "best_student_model")
    os.makedirs(best_teacher_dir, exist_ok=True)
    os.makedirs(best_student_dir, exist_ok=True)

    trainer.model.save_pretrained(best_teacher_dir) # type: ignore
    trainer.student_model.save_pretrained(best_student_dir) # type: ignore
    tokenizer.save_pretrained(best_teacher_dir)
    tokenizer.save_pretrained(best_student_dir)

    print(f"[OK] Teacher salvo em:  {best_teacher_dir}")
    print(f"[OK] Student salvo em:  {best_student_dir}")

    # ========================================================
    # Avaliação dos dois modelos (como é feito no paper)
    # ========================================================
    print("\n=== Avaliação Final (Teacher vs Student) ===")
    teacher_loss, teacher_ppl = evaluate_model(trainer.model, tokenizer, eval_dataset)
    student_loss, student_ppl = evaluate_model(trainer.student_model, tokenizer, eval_dataset)

    print(f"Teacher -> Loss: {teacher_loss:.4f} | Perplexity: {teacher_ppl:.2f}")
    print(f"Student -> Loss: {student_loss:.4f} | Perplexity: {student_ppl:.2f}")

    print("\nTreinamento e avaliação concluídos com sucesso!")


if __name__ == "__main__":
    main()
