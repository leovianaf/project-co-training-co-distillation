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
    save_strategy="no", # Não vamos salvar checkpoints neste teste
  )

  # Instanciar e Executar o Trainer Customizado
  trainer = CTCDTrainer(
    model=teacher_model,      # O modelo principal (professor)
    student_model=student_model, # Modelo estudante
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
  )

  print("Iniciando o co-treinamento...")
  trainer.train()
  print("Treinamento concluído com sucesso!")

if __name__ == '__main__':
    main()