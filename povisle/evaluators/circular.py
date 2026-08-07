from collections.abc import Iterable
from dataclasses import asdict
from itertools import permutations
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm

from povisle.backends.base import BaseBackend
from povisle.configs import PostprocessingConfig
from povisle.evaluators.base import BaseEvaluator
from povisle.postprocess import postprocess
from povisle.tasks.base import BaseTask, ScoringMethod
from povisle.tasks.mcq import choice_map
from povisle.utils import circular_rotations


CircularMode = Literal["circular", "all"]


class CircularEvaluator(BaseEvaluator):
    def __init__(
        self,
        task: BaseTask,
        backend: BaseBackend,
        run_metadata: dict[str, Any],
        postprocessing_config: PostprocessingConfig | None = None,
        mode: CircularMode = "circular",
        no_image: bool = False,
        no_question: bool = False,
    ):
        super().__init__(task, backend)
        self.run_metadata = run_metadata
        self.postprocessing_config = postprocessing_config
        self.mode = mode
        self.no_image = no_image
        self.no_question = no_question

    def evaluate(self, dataset: Iterable[dict[str, Any]]) -> pd.DataFrame:
        records = []
        for original_index, row in enumerate(dataset):
            row = dict(row)
            records.extend(self._create_variant_records(row, original_index))

        if not records:
            return pd.DataFrame()

        generations = self.backend.generate_batch(
            [
                {
                    "image": record["image"],
                    "prompt": record["prompt"],
                    "task": self.task.name,
                    "row": record["variant_row"],
                }
                for record in tqdm(records, desc=f"Preparing circular {self.task.name}")
            ]
        )

        groups: dict[int, dict[str, Any]] = {}
        for record, (raw_prediction, error, metadata) in tqdm(
            zip(records, generations, strict=True),
            total=len(records),
            desc=f"Scoring circular {self.task.name}",
        ):
            variant_row, prompt = record["variant_row"], record["prompt"]
            processed_prediction = postprocess(raw_prediction, self.postprocessing_config)

            score = self.task.score_prediction(processed_prediction, variant_row)
            is_correct = score.score == 1.0

            group = groups.setdefault(
                record["original_index"],
                {
                    "original_row": record["original_row"],
                    "predictions": [],
                    "variants_correct": 0,
                },
            )
            group["variants_correct"] += is_correct
            group["predictions"].append(
                {
                    "circular_option_order": "".join(record["option_order"]),
                    "circular_variant_answer": variant_row["answer"],
                    "prompt": prompt,
                    "raw_prediction": raw_prediction,
                    "processed_prediction": processed_prediction,
                    "error": error,
                    "metadata": metadata,
                    **asdict(score),
                }
            )

        rows = []
        for group in groups.values():
            variants_total = len(group["predictions"])
            variants_correct = int(group["variants_correct"])
            all_correct = variants_correct == variants_total

            rows.append(
                {
                    "run_id": self.run_metadata["run_id"],
                    "model_id": self.run_metadata["model_id"],
                    "dataset_id": self.run_metadata["dataset_id"],
                    "dataset_revision": self.run_metadata["dataset_revision"],
                    "backend": self.run_metadata["backend"],
                    "task": self.task.name,
                    **group["original_row"],
                    "score": 1.0 if all_correct else 0.0,
                    "scoring_method": ScoringMethod.CIRCULAR,
                    "circular_enabled": True,
                    "circular_mode": self.mode,
                    "circular_variants_total": variants_total,
                    "circular_variants_correct": variants_correct,
                    "circular_variants_all_correct": all_correct,
                    "predictions": group["predictions"],
                }
            )

        return pd.DataFrame(rows)

    def _create_variant_records(self, row: dict[str, Any], original_index: int) -> list[dict[str, Any]]:
        labels = tuple(choice_map(row).keys())
        image = row.pop("image")
        records = []

        for option_order in self._option_orders(labels):
            variant_row = self._variant_row(row, labels, option_order)
            records.append(
                {
                    "image": None if self.no_image else image,
                    "prompt": self.task.build_prompt(variant_row, no_question=self.no_question),
                    "original_row": row,
                    "variant_row": variant_row,
                    "original_index": original_index,
                    "labels": labels,
                    "option_order": option_order,
                }
            )

        return records

    def _option_orders(self, labels: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
        if self.mode == "circular":
            return circular_rotations(labels)
        if self.mode == "all":
            return permutations(labels)
        raise ValueError(f"Unsupported circular mode: {self.mode}")

    def _variant_row(
        self, row: dict[str, Any], labels: tuple[str, ...], option_order: tuple[str, ...]
    ) -> dict[str, Any]:
        variant = dict(row)
        for variant_label, original_label in zip(labels, option_order, strict=True):
            variant[variant_label] = row[original_label]

        original_answer = str(row["answer"]).strip().upper()
        variant["answer"] = labels[option_order.index(original_answer)]
        return variant

    def _to_original_label(
        self, parsed_prediction: str | None, option_order: tuple[str, ...], labels: tuple[str, ...]
    ) -> str | None:
        if parsed_prediction is None:
            return None

        parsed = str(parsed_prediction).strip().upper()
        if parsed not in labels:
            return None

        return option_order[labels.index(parsed)]
