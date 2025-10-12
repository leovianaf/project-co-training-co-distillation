import os
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

def main():
    project_dir = os.path.join(os.path.dirname(__file__), '..')
    processed_data_path = os.path.join(project_dir, 'data', 'processed')

    print("Carregando subconjuntos locais...")
    bookcorpus_local = load_from_disk(os.path.join(processed_data_path, 'bookcorpus_subset'))
    wiki_local = load_from_disk(os.path.join(processed_data_path, 'wiki_subset'))

    # Garante que os datasets carregados são DatasetDict e seleciona a partição 'train'
    if isinstance(bookcorpus_local, dict):
      bookcorpus_local = bookcorpus_local['train']
    if isinstance(wiki_local, dict):
      wiki_local = wiki_local['train']

    print("Limpando e combinando datasets...")
    # Remove colunas extras da Wikipedia para que os datasets possam ser concatenados
    wiki_local_cleaned = wiki_local.remove_columns(['id', 'url', 'title'])

    # Combina os dois datasets
    combined_ds = concatenate_datasets([bookcorpus_local, wiki_local_cleaned])
    print(f"Dataset combinado criado com {len(combined_ds)} exemplos.")

    # Carrega o tokenizador padrão do BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
      # O tokenizador processa o texto para o formato que o BERT espera
      return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    print("Tokenizando o dataset (isso pode levar um momento)...")
    tokenized_ds = combined_ds.map(tokenize_function, batched=True, remove_columns=['text'])

    # Salva o dataset final e tokenizado
    output_path = os.path.join(processed_data_path, 'tokenized_sample_dataset')
    print(f"Salvando dataset tokenizado em: {output_path}")
    tokenized_ds.save_to_disk(output_path)

    print("Processamento de dados concluído com sucesso!")


if __name__ == '__main__':
    main()