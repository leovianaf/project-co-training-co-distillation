import numpy as np
import evaluate

TASK_CONFIGS = {
    "stsb": {
        "num_labels": 1,
        "problem_type": "regression",
        "metric_name": "pearsonr",
        "greater_is_better": True,
    },
    "mrpc": {
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "metric_name": "f1",
        "greater_is_better": True,
    },
    "rte": {
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "metric_name": "accuracy",
        "greater_is_better": True,
    },
}

TASK_CONFIGS_PT = {
    "stsb": {
        "dataset_name": "assin2",
        "metric_name": "pearsonr",
        "num_labels": 1,
        "problem_type": "regression",
        "label_column": "relatedness_score",
        "greater_is_better": True,
    },
    "rte": {
        "dataset_name": "assin2",
        "metric_name": "accuracy",
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "label_column": "entailment_judgment",
        "greater_is_better": True,
    },
}

def get_compute_metrics_fn(task_name):
    if task_name == "stsb":
        metric = evaluate.combine(["pearsonr", "spearmanr"])
    elif task_name == "mrpc":
        metric = evaluate.combine(["f1", "accuracy"])
    else:  # rte
        metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task_name == "stsb":
            # Regressão → logits têm shape (N, 1)
            preds = np.squeeze(logits)
            return metric.compute(predictions=preds, references=labels)
        else:
            # Classificação → logits têm shape (N, C)
            preds = np.argmax(logits, axis=-1)
            return metric.compute(predictions=preds, references=labels)

    return compute_metrics
