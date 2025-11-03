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
#  Funções auxiliares
# ============================================================

def load_metrics(model_type, task_name):
    path = os.path.join(RESULTS_DIR, model_type, task_name, "all_results.json")
    if not os.path.exists(path):
        print(f"[AVISO] {path} não encontrado.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def extract_main_metric(task_name, metrics):
    """Seleciona a métrica principal de cada task (igual ao paper)."""
    if metrics is None:
        return None
    if task_name == "stsb":
        pearson = metrics.get("eval_pearsonr", 0)
        spearman = metrics.get("eval_spearmanr", 0)
        return (pearson + spearman) / 2
    elif task_name in ["mrpc", "rte"]:
        return metrics.get("eval_accuracy", 0)
    return 0

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
#  Estilo unificado
# ============================================================
def apply_professional_style():
    """Define estilo visual padrão para todos os gráficos."""
    sns.set_theme(
        style="white", 
        context="paper",
        font_scale=1.1
    )
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })

# Paleta única
PALETTE = {
    "Teacher": "#4878CF",  # azul técnico
    "Student": "#E6B800"   # dourado elegante
}
ACCENT_COLOR = "#345995"   # azul profundo para a linha GAP

# ============================================================
#  1. Gráfico por Task
# ============================================================
def plot_task_comparison(df):
    if df.empty:
        print("Nenhum dado encontrado para plotar.")
        return

    apply_professional_style()
    plt.figure(figsize=(6, 4))

    ax = sns.barplot(
        data=df, x="Task", y="Score", hue="Model",
        palette=PALETTE, width=0.6, edgecolor="black", linewidth=0.6
    )

    # === Configuração dos eixos ===
    ax.set_ylim(0, 1)
    ax.set_title("Performance Comparativa – Teacher vs Student (GLUE)",
                 fontsize=10, weight="bold", pad=10)
    ax.set_ylabel("Score (métrica principal)", fontsize=9)
    ax.set_xlabel("Task", fontsize=9)
    ax.legend(title="", fontsize=8, loc="upper left", frameon=False)
    sns.despine(top=True, right=True)

    # === Adiciona rótulos de porcentagem sobre cada barra ===
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,  # centro da barra
                height + 0.015,                # ligeiramente acima
                f"{height*100:.1f}%",          # converte para %
                ha="center", va="bottom",
                fontsize=8, color="#333333"
            )

    plt.tight_layout()

    # === Salva e mostra ===
    save_path = os.path.join(OUT_DIR, "performance_by_task.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[OK] Figura salva em: {save_path}")


# ============================================================
#  2. Gráfico de Média Geral
# ============================================================
def plot_overall_comparison(df):
    if df.empty:
        print("Nenhum dado encontrado para plotar.")
        return

    mean_df = (
        df.groupby("Model", as_index=False)["Score"]
        .mean()
        .sort_values("Model")
    )

    apply_professional_style()
    plt.figure(figsize=(3.5, 3))

    ax = sns.barplot(
        data=mean_df,
        x="Model",
        y="Score",
        palette=PALETTE,
        width=0.5,
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score médio (todas as tasks)", fontsize=9)
    ax.set_title("Média Geral – Teacher vs Student", fontsize=10, weight="bold")
    sns.despine(top=True, right=True)

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "performance_overall.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Figura salva em: {save_path}")

    plt.show()


# ============================================================
#  3. Gráfico de GAP (%)
# ============================================================
def plot_gap_comparison(df):
    if df.empty:
        print("Nenhum dado encontrado para plotar.")
        return

    apply_professional_style()
    pivot = df.pivot(index="Task", columns="Model", values="Score").reset_index()
    pivot["GAP_%"] = (pivot["Student"] - pivot["Teacher"]) * 100

    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    # Barras
    pivot_melted = pivot.melt(id_vars=["Task"], value_vars=["Teacher", "Student"],
                              var_name="Model", value_name="Score")

    sns.barplot(
        data=pivot_melted, x="Task", y="Score", hue="Model",
        ax=ax1, palette=PALETTE, width=0.6, edgecolor="black", linewidth=0.6
    )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score (métrica principal)", fontsize=9)
    ax1.set_xlabel("Task", fontsize=9)
    ax1.set_title("Teacher vs Student — Performance GAP (%)", fontsize=10, weight="bold")

    # Linha GAP
    ax2 = ax1.twinx()
    ax2.plot(pivot["Task"], pivot["GAP_%"], color=ACCENT_COLOR,
             marker="o", linestyle="--", linewidth=1.8, markersize=5)
    ax2.set_ylabel("Ganho do Student sobre Teacher (%)", color=ACCENT_COLOR, fontsize=8)
    ax2.tick_params(axis='y', labelcolor=ACCENT_COLOR, labelsize=8)
    ax2.set_ylim(0, max(pivot["GAP_%"]) * 1.25)

    for x, y in zip(pivot["Task"], pivot["GAP_%"]):
        ax2.text(x, y + 0.1, f"{y:+.2f}%", color=ACCENT_COLOR,
                 ha="center", fontsize=8, fontweight="medium")

    ax1.legend(title="", fontsize=8, loc="upper left", frameon=False)
    sns.despine(top=True, right=True)
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, "performance_gap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[OK] Figura salva em: {save_path}")

# ============================================================
#  Execução
# ============================================================
if __name__ == "__main__":
    df = gather_results()
    print(df)

    plot_task_comparison(df)
    plot_overall_comparison(df)
    plot_gap_comparison(df)
