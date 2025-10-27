import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================
#  Configurações de Diretórios
# ============================================================
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_DIR, "models", "finetuned")
OUT_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  Função para carregar métricas
# ============================================================

def load_metrics(model_type, task_name):
    path = os.path.join(RESULTS_DIR, model_type, task_name, "all_results.json")
    if not os.path.exists(path):
        print(f"[AVISO] {path} não encontrado.")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data

# ============================================================
#  Normalizar métricas relevantes (para comparação justa)
# ============================================================

def extract_main_metric(task_name, metrics):
    """Seleciona a métrica principal de cada task (igual ao paper)."""
    if metrics is None:
        return None
    if task_name == "stsb":
        # Usar média de pearsonr + spearmanr
        pearson = metrics.get("eval_pearsonr", 0)
        spearman = metrics.get("eval_spearmanr", 0)
        return (pearson + spearman) / 2
    elif task_name in ["mrpc", "rte"]:
        return metrics.get("eval_accuracy", 0)
    return 0

# ============================================================
#  Gerar DataFrame consolidado (Teacher × Student)
# ============================================================

def gather_results():
    rows = []
    for task in ["stsb", "mrpc", "rte"]:
        for model_type in ["teacher", "student"]:
            metrics = load_metrics(model_type, task)
            if metrics:
                score = extract_main_metric(task, metrics)
                rows.append({
                    "Task": task.upper(),
                    "Model": model_type.capitalize(),
                    "Score": score
                })
    return pd.DataFrame(rows)

# ============================================================
#  Plot estilo do paper
# ============================================================

def plot_comparison(df):
    if df.empty:
        print("Nenhum dado encontrado para plotar.")
        return

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 5))

    # Gráfico de barras lado a lado
    sns.barplot(
        data=df,
        x="Task",
        y="Score",
        hue="Model",
        palette={"Teacher": "#4C72B0", "Student": "#DD8452"},
        width=0.6
    )

    plt.ylim(0, 1)
    plt.title("Performance Comparativa – Teacher vs Student (GLUE)")
    plt.ylabel("Score (principal metric)")
    plt.xlabel("Task")
    plt.legend(title="")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, "performance_comparison_paper_style.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[OK] Figura salva em: {save_path}")

# ============================================================
#  Execução
# ============================================================

if __name__ == "__main__":
    df = gather_results()
    print(df)
    plot_comparison(df)
