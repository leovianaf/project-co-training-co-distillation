import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================
#  Configurações gerais
# ============================================================
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_DIR, "models", "finetuned")
OUT_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Paleta idêntica ao gráfico original
PALETTE = {
    "Teacher": "#4878CF",
    "Student": "#E6B800",
}
ACCENT_COLOR = "#345995"


# ============================================================
#  Funções auxiliares
# ============================================================

def load_metrics(language, model_type, task_name):
    """
    Carrega all_results.json no novo caminho:
    models/finetuned/<lang>/<model>/<task>/all_results.json
    """
    file_path = os.path.join(
        RESULTS_DIR,
        language,
        model_type,
        task_name,
        "all_results.json"
    )

    if not os.path.exists(file_path):
        print(f"[AVISO] {file_path} não encontrado.")
        return None

    with open(file_path, "r") as f:
        return json.load(f)


def extract_metric(task_name, metrics, language):
    """
    Extrai métrica principal dependendo da task e idioma.
    """
    if metrics is None:
        return None

    # Regressão: STS-B (EN) e ASSIN2-STSB (PT)
    if task_name == "stsb":
        p = metrics.get("eval_pearsonr")
        s = metrics.get("eval_spearmanr")
        if p is None or s is None:
            return None
        return (p + s) / 2

    # Classificação binária: MRPC (EN), RTE (EN/PT)
    return metrics.get("eval_accuracy")


def gather_results(language):
    """
    Monta tabela final com: Task | Model | Score
    """

    if language == "en":
        TASKS = ["stsb", "mrpc", "rte"]
    else:  # pt
        TASKS = ["stsb", "rte"]

    rows = []

    for task in TASKS:
        for model_type in ["teacher", "student"]:
            metrics = load_metrics(language, model_type, task)
            score = extract_metric(task, metrics, language)

            if score is not None:
                rows.append({
                    "Task": task.upper(),
                    "Model": model_type.capitalize(),
                    "Score": score
                })

    return pd.DataFrame(rows)


# ============================================================
#  Estilo unificado
# ============================================================
def apply_professional_style():
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#333",
        "xtick.color": "#333",
        "ytick.color": "#333",
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })


# ============================================================
#  1. Performance por Task
# ============================================================
def plot_task_comparison(df, lang):
    if df.empty:
        print("[ERRO] DF vazio — nada para plotar.")
        return

    apply_professional_style()
    plt.figure(figsize=(6, 4))

    ax = sns.barplot(
        data=df, x="Task", y="Score", hue="Model",
        palette=PALETTE, width=0.6, edgecolor="black"
    )

    ax.set_ylim(0, 1)
    ax.set_title(f"Performance Comparativa – Teacher vs Student ({lang.upper()})",
                 fontsize=10, weight="bold")

    # labels acima das barras
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.015,
                f"{height*100:.1f}%",
                ha="center", fontsize=8
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{lang}_by_task.png"), dpi=300)
    plt.show()


# ============================================================
#  2. Média Geral
# ============================================================
def plot_overall(df, lang):
    if df.empty:
        print("[ERRO] DF vazio — nada para plotar.")
        return

    mean_df = df.groupby("Model", as_index=False)["Score"].mean()

    apply_professional_style()
    plt.figure(figsize=(4, 3))

    ax = sns.barplot(
        data=mean_df, x="Model", y="Score",
        palette=PALETTE, edgecolor="black"
    )

    ax.set_ylim(0, 1)
    ax.set_title(f"Média Geral – Teacher vs Student ({lang.upper()})")

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2,
            height + 0.02,
            f"{height:.3f}",
            ha="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{lang}_overall.png"), dpi=300)
    plt.show()


# ============================================================
#  3. GAP (%)
# ============================================================
def plot_gap(df, lang):
    if df.empty:
        print("[ERRO] DF vazio — nada para plotar.")
        return

    pivot = df.pivot(index="Task", columns="Model", values="Score").reset_index()
    pivot["GAP_%"] = (pivot["Student"] - pivot["Teacher"]) * 100

    apply_professional_style()
    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    melted = pivot.melt(id_vars=["Task"], value_vars=["Teacher", "Student"],
                        var_name="Model", value_name="Score")

    sns.barplot(
        data=melted, x="Task", y="Score", hue="Model",
        palette=PALETTE, ax=ax1, edgecolor="black"
    )

    ax1.set_ylim(0, 1)
    ax1.set_title(f"Teacher vs Student — Performance GAP (%) ({lang.upper()})")

    ax2 = ax1.twinx()
    ax2.plot(
        pivot["Task"], pivot["GAP_%"],
        color=ACCENT_COLOR, marker="o", linestyle="--"
    )

    for x, y in zip(pivot["Task"], pivot["GAP_%"]):
        ax2.text(x, y + 0.1, f"{y:+.2f}%", color=ACCENT_COLOR, ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{lang}_gap.png"), dpi=300)
    plt.show()


# ============================================================
#  Execução
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, choices=["en", "pt"])
    args = parser.parse_args()

    df = gather_results(args.language)
    print(df)

    plot_task_comparison(df, args.language)
    plot_overall(df, args.language)
    plot_gap(df, args.language)
