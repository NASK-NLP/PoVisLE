from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

from povisle.tasks import ParsingStatus
from povisle.tasks.mcq import LABELS as MCQ_LABELS
from povisle.tasks.yn import LABELS as YN_LABELS
from povisle.tasks.yn import canonical_yes_no


CLOSED_TASK_LABELS = {
    "mcq": MCQ_LABELS,
    "yn": YN_LABELS,
}


def calculate_metrics(results_by_task: dict[str, pd.DataFrame], evaluation_mode: str = "default") -> dict[str, Any]:
    by_task = {task_name: calculate_task_metrics(frame) for task_name, frame in results_by_task.items()}
    all_results = pd.concat(list(results_by_task.values()), ignore_index=True) if results_by_task else pd.DataFrame()
    overall = calculate_mixed_metrics(all_results)
    overall.update(calculate_cross_task_metrics(by_task))

    return {
        "overall": overall,
        "by_task": by_task,
        "by_category": calculate_category_metrics(all_results),
        "by_category_and_subcategory": calculate_category_and_subcategory_metrics(all_results),
        "by_image_source": calculate_image_source_metrics(all_results),
        "by_task_and_category": {
            task_name: calculate_task_category_metrics(frame) for task_name, frame in results_by_task.items()
        },
        "by_task_and_image_source": {
            task_name: calculate_task_image_source_metrics(frame) for task_name, frame in results_by_task.items()
        },
        "confusion_matrices": (
            calculate_confusion_matrices(results_by_task) if evaluation_mode == "default" else {}
        ),
    }


def calculate_cross_task_metrics(by_task: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    accuracies = [
        metrics.get("accuracy")
        for metrics in by_task.values()
        if metrics.get("accuracy") is not None
    ]

    return {
        "macro_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else None,
    }


def leaderboard_row(run_metadata: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    row = {**run_metadata, **{f"overall_{key}": value for key, value in summary["overall"].items()}}
    for task_name, metrics in summary["by_task"].items():
        for key, value in metrics.items():
            row[f"{task_name}_{key}"] = value
    return row


def calculate_task_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "total": 0,
            "accuracy": None,
            "parse_rate": None,
            "runtime_failure_rate": None,
        }

    predictions = get_all_predictions(frame)
    parsed_statuses = [
        prediction.get("parsing_status")
        for prediction in predictions
        if prediction.get("parsing_status") is not None
    ]
    parse_rate = (
        float(sum(status == ParsingStatus.PARSED for status in parsed_statuses) / len(parsed_statuses))
        if parsed_statuses
        else None
    )
    errors = [prediction.get("error") for prediction in predictions]

    metrics = {
        "total": int(len(frame)),
        "accuracy": float((frame["score"].fillna(0.0) == 1.0).mean()),
        "parse_rate": parse_rate,
        "runtime_failure_rate": (
            float(sum(error is not None and not pd.isna(error) for error in errors) / len(errors))
            if errors
            else 0.0
        ),
    }

    labels = closed_labels_for_frame(frame)
    if labels == YN_LABELS and uses_single_prediction(frame):
        metrics["balanced_accuracy"] = calculate_balanced_accuracy(frame, labels)

    return metrics


def calculate_mixed_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = calculate_task_metrics(frame)
    accuracy = metrics.pop("accuracy")
    metrics["micro_accuracy"] = accuracy
    metrics["macro_accuracy"] = calculate_macro_accuracy(frame)
    return metrics


def calculate_macro_accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty or "task" not in frame:
        return None

    task_accuracies = [
        calculate_task_metrics(task_frame)["accuracy"]
        for _, task_frame in frame.groupby("task", sort=False, dropna=False)
    ]
    task_accuracies = [accuracy for accuracy in task_accuracies if accuracy is not None]
    return float(sum(task_accuracies) / len(task_accuracies)) if task_accuracies else None


def calculate_category_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "category" not in frame:
        return {}

    records = {}
    for category, category_frame in frame.groupby("category", dropna=False):
        metrics = calculate_mixed_metrics(category_frame)
        category_name = "unknown" if pd.isna(category) else str(category)
        records[category_name] = metrics

    return dict(sorted(records.items()))


def calculate_category_and_subcategory_metrics(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    if frame.empty or "category" not in frame or "subcategory" not in frame:
        return {}

    records = {}
    for category, category_frame in frame.groupby("category", dropna=False):
        category_name = "unknown" if pd.isna(category) else str(category)
        subcategory_records = {}
        for subcategory, subcategory_frame in category_frame.groupby("subcategory", dropna=False):
            subcategory_name = "unknown" if pd.isna(subcategory) else str(subcategory)
            subcategory_records[subcategory_name] = calculate_mixed_metrics(subcategory_frame)
        records[category_name] = dict(sorted(subcategory_records.items()))

    return dict(sorted(records.items()))


def calculate_task_category_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "category" not in frame:
        return {}

    records = {}
    for category, category_frame in frame.groupby("category", dropna=False):
        category_name = "unknown" if pd.isna(category) else str(category)
        records[category_name] = calculate_task_metrics(category_frame)

    return dict(sorted(records.items()))


def calculate_image_source_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "image_source" not in frame:
        return {}

    records = {}
    for image_source, image_source_frame in frame.groupby("image_source", dropna=False):
        image_source_name = "unknown" if pd.isna(image_source) else str(image_source)
        records[image_source_name] = calculate_mixed_metrics(image_source_frame)

    return dict(sorted(records.items()))


def calculate_task_image_source_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "image_source" not in frame:
        return {}

    records = {}
    for image_source, image_source_frame in frame.groupby("image_source", dropna=False):
        image_source_name = "unknown" if pd.isna(image_source) else str(image_source)
        records[image_source_name] = calculate_task_metrics(image_source_frame)

    return dict(sorted(records.items()))


def calculate_confusion_matrices(results_by_task: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    matrices = {}
    for task_name, labels in CLOSED_TASK_LABELS.items():
        frame = results_by_task.get(task_name)
        if frame is None or frame.empty:
            continue

        special_labels = ("__unparsed__", "__error__")
        all_labels = [*labels, *special_labels]
        gold = closed_gold_labels(frame, task_name)
        predicted = frame.apply(_confusion_prediction_label, axis=1)

        matrices[task_name] = {
            "labels": all_labels,
            "matrix": confusion_matrix(gold, predicted, labels=all_labels).astype(int).tolist(),
        }

    return matrices


def _confusion_prediction_label(row: pd.Series) -> str:
    prediction = first_prediction(row)
    if prediction.get("error") is not None and not pd.isna(prediction.get("error")):
        return "__error__"
    if prediction.get("parsing_status") != ParsingStatus.PARSED:
        return "__unparsed__"
    return str(prediction.get("parsed_prediction")).strip()


def closed_labels_for_frame(frame: pd.DataFrame) -> tuple[str, ...] | None:
    if "task" not in frame:
        return None

    task_names = set(frame["task"].dropna().unique())
    if len(task_names) != 1:
        return None

    return CLOSED_TASK_LABELS.get(next(iter(task_names)))


def closed_gold_labels(frame: pd.DataFrame, task_name: str) -> pd.Series:
    gold = frame["answer"].fillna("").astype(str).str.strip()
    if task_name == "yn":
        return gold.apply(canonical_yes_no)
    return gold


def calculate_balanced_accuracy(frame: pd.DataFrame, labels: tuple[str, ...]) -> float:
    gold = closed_gold_labels(frame, "yn")
    predicted = frame.apply(lambda row: first_prediction(row).get("parsed_prediction") or "__unparsed__", axis=1)
    predicted = predicted.astype(str).str.strip()
    return float(recall_score(gold, predicted, labels=labels, average="macro", zero_division=0))


def get_all_predictions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        prediction
        for predictions in frame["predictions"].dropna()
        if isinstance(predictions, list)
        for prediction in predictions
        if isinstance(prediction, dict)
    ]


def first_prediction(row: pd.Series) -> dict[str, Any]:
    predictions = row.get("predictions")
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
        return predictions[0]

    return {}


def uses_single_prediction(frame: pd.DataFrame) -> bool:
    return bool(
        frame["predictions"].apply(
            lambda predictions: isinstance(predictions, list) and len(predictions) == 1
        ).all()
    )
