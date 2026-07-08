import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from povisle.backends import create_backend
from povisle.configs import ModelConfig
from povisle.evaluators import CircularEvaluator, DefaultEvaluator
from povisle.logger import get_logger
from povisle.metrics import calculate_metrics
from povisle.system import collect_system_metadata
from povisle.tasks import get_tasks
from povisle.utils import format_percent, load_task_dataset, make_run_id, write_json

logger = get_logger(__name__)


def save_run(
    results_dir: Path,
    results_by_task: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
) -> pd.DataFrame:
    run_dir = results_dir / "runs" / run_metadata["run_id"]

    write_json(run_dir / "run_metadata.json", run_metadata)
    write_json(run_dir / "summary_metrics.json", metrics)

    predictions = pd.concat(list(results_by_task.values()), ignore_index=True)
    predictions.to_json(run_dir / "all_predictions.json", orient="records", force_ascii=False, indent=2)

    return predictions


def update_leaderboard(
    results_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
) -> Path:
    model_id = run_metadata["model_id"]
    fallback_org, _, fallback_name = model_id.partition("/")
    if not fallback_name:
        fallback_org = "local"
        fallback_name = model_id

    org = run_metadata.get("org") or fallback_org
    model_name = run_metadata.get("model_name") or fallback_name

    dataset_version = run_metadata["dataset_revision"]
    split = run_metadata["split"]
    evaluation_mode = run_metadata.get("evaluation_mode", "default")
    leaderboard_dir = results_dir / "leaderboard" / org / model_name / dataset_version / split / evaluation_mode

    results = {
        "run_metadata": run_metadata,
        "metrics": metrics,
    }
    write_json(leaderboard_dir / "results.json", results)

    predictions.to_json(leaderboard_dir / "predictions.json", orient="records", force_ascii=False, indent=2)

    return leaderboard_dir


def push_evaluation_outputs_to_hub(
    leaderboard_dir: Path,
    run_dir: Path,
    results_dir: Path,
    repo_id: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for folder_path in (leaderboard_dir, run_dir):
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=folder_path.relative_to(results_dir).as_posix(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLMs on the povisle benchmark.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-id", type=str, default="NASK-PIB/PoVisLE")
    parser.add_argument("--dataset-revision", type=str, default="v1.0.1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--tasks", nargs="+", choices=["all", "mcq", "yn", "open"], default=["all"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-circural", action="store_true")
    parser.add_argument("--circural-mode", choices=["circular", "all"], default="circular")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-question", action="store_true")
    parser.add_argument("--hf-push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    args = parser.parse_args()

    if args.hf_push_to_hub and not args.hf_repo_id:
        parser.error("--hf-repo-id is required when --hf-push-to-hub is set")

    load_dotenv()

    logger.info("Loading model configuration from %s", args.model_config)
    model_config = ModelConfig.from_yaml(args.model_config)

    logger.info("Selected tasks: %s", args.tasks)
    tasks = get_tasks(args.tasks)

    run_id = make_run_id(model_config)
    logger.info("Run ID: %s", run_id)

    use_circular = args.use_circural and any(task.name == "mcq" for task in tasks)
    if args.use_circural and not use_circular:
        logger.warning("--use-circural was set, but no mcq task was selected. Circular evaluation will be skipped.")

    evaluation_mode = f"circular_{args.circural_mode}" if use_circular else "default"
    if args.no_image:
        evaluation_mode = "no_image" if evaluation_mode == "default" else f"{evaluation_mode}_no_image"
    if args.no_question:
        evaluation_mode = "no_question" if evaluation_mode == "default" else f"{evaluation_mode}_no_question"
    run_metadata = {
        "run_id": run_id,
        "model_id": model_config.model_id,
        "org": model_config.org,
        "model_name": model_config.name,
        "model_family": model_config.model_family,
        "model_type": model_config.model_type,
        "model_size": model_config.model_size,
        "backend": model_config.backend,
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "tasks": [task.name for task in tasks],
        "limit": args.limit,
        "evaluation_mode": evaluation_mode,
        "no_image": args.no_image,
        "no_question": args.no_question,
        "circular_enabled": use_circular,
        "circular_mode": args.circural_mode if use_circular else None,
        "preprocessing": asdict(model_config.preprocessing),
        "postprocessing": asdict(model_config.postprocessing),
        "model_config": asdict(model_config),
        "system": collect_system_metadata(),
    }

    results_by_task = {}
    with create_backend(model_config) as backend:
        for task in tasks:
            task_subset = load_task_dataset(
                dataset_id=args.dataset_id,
                subset=task.name,
                split=args.split,
                revision=args.dataset_revision,
                limit=args.limit,
            )
            if use_circular and task.name == "mcq":
                evaluator = CircularEvaluator(
                    task=task,
                    backend=backend,
                    run_metadata=run_metadata,
                    postprocessing_config=model_config.postprocessing,
                    mode=args.circural_mode,
                    no_image=args.no_image,
                    no_question=args.no_question,
                )
            else:
                evaluator = DefaultEvaluator(
                    task=task,
                    backend=backend,
                    run_metadata=run_metadata,
                    postprocessing_config=model_config.postprocessing,
                    no_image=args.no_image,
                    no_question=args.no_question,
                )
            results_by_task[task.name] = evaluator.evaluate(task_subset)

    logger.info("Calculating metrics...")
    metrics = calculate_metrics(results_by_task, evaluation_mode=evaluation_mode)
    logger.info("Metrics:\n%s", pd.DataFrame(
        [
            ("Overall Acc", format_percent(metrics["overall"]["strict_accuracy"])),
            ("MC Strict Acc", format_percent(metrics["by_task"].get("mcq", {}).get("strict_accuracy"))),
            ("Yes/No Strict Acc", format_percent(metrics["by_task"].get("yn", {}).get("strict_accuracy"))),
            ("Open Strict Acc", format_percent(metrics["by_task"].get("open", {}).get("strict_accuracy"))),
            ("Parse rate", format_percent(metrics["overall"]["parse_rate"])),
        ],
        columns=["Metric", "Value"],
    ).to_markdown(index=False))

    logger.info("Saving run outputs to %s", args.results_dir / "runs" / run_id)
    predictions = save_run(args.results_dir, results_by_task, metrics, run_metadata)
    run_dir = args.results_dir / "runs" / run_id

    leaderboard_dir = update_leaderboard(args.results_dir, predictions, metrics, run_metadata)
    logger.info("Updated leaderboard entry at %s", leaderboard_dir)

    if args.hf_push_to_hub:
        assert args.hf_repo_id is not None, "HF repo ID must be provided when pushing to hub"
        logger.info("Pushing leaderboard entry and run artifacts to Hugging Face dataset repo %s", args.hf_repo_id)
        push_evaluation_outputs_to_hub(
            leaderboard_dir=leaderboard_dir,
            run_dir=run_dir,
            results_dir=args.results_dir,
            repo_id=args.hf_repo_id,
        )
        logger.info("Pushed leaderboard entry and run artifacts to Hugging Face.")

    logger.info("Evaluation completed successfully.")
