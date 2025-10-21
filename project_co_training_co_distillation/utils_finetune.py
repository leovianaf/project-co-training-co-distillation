import evaluate

# Mapeia tasks para suas métricas e configurações de modelo
TASK_CONFIGS = {
  "stsb": {
    "metric_name": "spearmanr",
    "num_labels": 1,
    "problem_type": "regression",
    "greater_is_better": True,
  },
  "mrpc": {
    "metric_name": "accuracy",
    "num_labels": 2,
    "problem_type": "single_label_classification",
    "greater_is_better": True,
  },
  "rte": {
    "metric_name": "accuracy",
    "num_labels": 2,
    "problem_type": "single_label_classification",
    "greater_is_better": True,
  },
}

def get_compute_metrics_fn(task_name):
  """Retorna a função compute_metrics correta para a tarefa."""
  metric = evaluate.load("glue", task_name)

  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task_name == "stsb":
      predictions = predictions.squeeze()
    return metric.compute(predictions=predictions, references=labels)

  return compute_metrics
