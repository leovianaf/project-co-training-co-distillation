import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
  BertConfig,
  AutoModelForMaskedLM,
  AutoTokenizer,
  DataCollatorForLanguageModeling,
  Trainer,
  TrainingArguments
)

# Custom Trainer para Co-Training e Co-Distillation
class CTCDTrainer(Trainer):
  def __init__(self, student_model, *args, **kwargs):
    # O Trainer principal será o professor, guardamos o estudante separadamente
    super().__init__(*args, **kwargs)
    self.student_model = student_model

  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # 'model' aqui é o professor
    teacher_model = model

    # Obter predições de ambos os modelos
    outputs_teacher = teacher_model(**inputs)
    # Move o estudante para o mesmo dispositivo que o professor
    outputs_student = self.student_model.to(self.args.device)(**inputs)

    # A loss de Masked Language Modeling (MLM) já vem calculada
    hard_loss_teacher = outputs_teacher.loss
    hard_loss_student = outputs_student.loss

    # Calcular a Soft Loss (KL Divergence)
    # A temperatura suaviza as distribuições de probabilidade
    temperature = 2.0

    # KL(teacher || student) -> para treinar o ESTUDANTE
    soft_loss_student = F.kl_div(
        input=F.log_softmax(outputs_student.logits / temperature, dim=-1),
        target=F.softmax(outputs_teacher.logits.detach() / temperature, dim=-1), # .detach() para o gradiente
        reduction='batchmean'
    ) * (temperature ** 2)

    # KL(student || teacher) -> para treinar o PROFESSOR
    soft_loss_teacher = F.kl_div(
        input=F.log_softmax(outputs_teacher.logits / temperature, dim=-1),
        target=F.softmax(outputs_student.logits.detach() / temperature, dim=-1), # .detach() para o gradiente
        reduction='batchmean'
    ) * (temperature ** 2)

    # Combinar as Losses usando pesos
    loss_teacher = hard_loss_teacher + soft_loss_teacher
    loss_student = hard_loss_student + soft_loss_student

    # A loss total é a soma das duas, para a retropropagação
    total_loss = loss_teacher + loss_student

    return (total_loss, outputs_teacher) if return_outputs else total_loss

def main():
  print(f"PyTorch tem acesso à GPU? {torch.cuda.is_available()}")

  project_dir = os.path.join(os.path.dirname(__file__), '..')
  tokenized_dataset_path = os.path.join(project_dir, 'data', 'processed', 'tokenized_sample_dataset')

  print(f"Carregando dataset de: {tokenized_dataset_path}")
  tokenized_ds = load_from_disk(tokenized_dataset_path)

  # Usaremos 10% dos dados para avaliação
  # Não é necessário importar o train_test_split explicitamente, pois é um método do Dataset
  split_datasets = tokenized_ds.train_test_split(test_size=0.1, seed=42) # type: ignore
  train_dataset = split_datasets['train']
  eval_dataset = split_datasets['test']
  print(f"Dataset dividido: {len(train_dataset)} para treino, {len(eval_dataset)} para avaliação.")

  print("Definindo e inicializando os modelos do zero...")
  teacher_config = BertConfig(num_hidden_layers=6, hidden_size=768, num_attention_heads=12)
  student_config = BertConfig(num_hidden_layers=4, hidden_size=768, num_attention_heads=12)

  teacher_model = AutoModelForMaskedLM.from_config(teacher_config)
  student_model = AutoModelForMaskedLM.from_config(student_config)

  # Configurar o Treinamento
  tokenizer_name = 'bert-base-uncased'
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
  )

  training_args = TrainingArguments(
    output_dir=os.path.join(project_dir, 'models', 'ctcd-sample-output'),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10,
    eval_strategy="epoch",          # Avalia a cada época
    save_strategy="epoch",          # Salva a cada época
    load_best_model_at_end=True,    # Carrega o melhor modelo no final
    metric_for_best_model="loss",   # Usa a 'loss' para decidir qual é o melhor
    greater_is_better=False,        # Menor loss é melhor
    save_total_limit=2,
  )

  # Instanciar e Executar o Trainer Customizado
  trainer = CTCDTrainer(
    model=teacher_model,
    student_model=student_model,
    args=training_args,
    train_dataset=train_dataset,  # Fornece o split de treino
    eval_dataset=eval_dataset,    # Fornece o split de avaliação
    data_collator=data_collator,
  )

  print("Iniciando o co-treinamento...")
  trainer.train()

  best_student_model_path = os.path.join(project_dir, 'models', 'best_student_model')
  trainer.student_model.save_pretrained(best_student_model_path)
  print(f"Melhor modelo estudante salvo em: {best_student_model_path}")

  print("Treinamento concluído com sucesso!")

if __name__ == '__main__':
    main()