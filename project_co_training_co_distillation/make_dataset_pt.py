import os
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

def prepare_pt_tasks():

    print("\n=== Preparando Datasets de Fine-Tuning (PT-BR) ===")

    # --- 1. ASSIN 2 (Substitui STS-B e RTE) ---
    print("⬇ Baixando ASSIN 2 (Similaridade e Entailment)...")
    try:
        assin2 = load_dataset("nilc-nlp/assin2")
        # Padronizar nomes das colunas para o Trainer
        assin2 = assin2.rename_columns({'premise': 'sentence1', 'hypothesis': 'sentence2'})

        save_path = os.path.join(TASKS_DIR, "assin2")
        assin2.save_to_disk(save_path) # type: ignore
        print(f"✅ ASSIN 2 salvo em: {save_path}")
    except Exception as e:
        print(f"❌ Erro ao baixar ASSIN 2: {e}")

def prepare_aroeira_pretraining():
    processed_path = os.path.join(DATA_DIR, 'processed')

    input_dataset_name = 'aroeira_subset_1k'
    input_path = os.path.join(processed_path, input_dataset_name)

    print(f"\n=== Preparando Dataset de Pré-treino (Aroeira) ===")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Pasta '{input_path}' não encontrada. Descompacte o arquivo do Colab em data/processed/.")

    print(f"Carregando dataset local de: {input_path}")
    aroeira_ds = load_from_disk(input_path)

    print(f"Exemplos carregados: {len(aroeira_ds)}")

    # Tokenizer em PORTUGUÊS. O 'uncased' do inglês é ruim para PT.
    model_checkpoint = "neuralmind/bert-base-portuguese-cased"
    print(f"Carregando tokenizer: {model_checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    print("Tokenizando o dataset...")
    # remove_columns=['text'] é importante para economizar espaço e evitar erros no Trainer
    tokenized_ds = aroeira_ds.map(tokenize_function, batched=True, remove_columns=['text'])

    output_path = os.path.join(processed_path, f'tokenized_{input_dataset_name}')
    print(f"Salvando dataset tokenizado em: {output_path}")
    tokenized_ds.save_to_disk(output_path)

    print("Processamento concluído com sucesso!")

if __name__ == '__main__':
    prepare_aroeira_pretraining()

    prepare_pt_tasks()