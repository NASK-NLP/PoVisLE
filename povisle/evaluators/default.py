from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import pandas as pd
from tqdm import tqdm

from povisle.backends.base import BaseBackend
from povisle.configs import PostprocessingConfig
from povisle.evaluators.base import BaseEvaluator
from povisle.postprocess import postprocess
from povisle.tasks.base import BaseTask


class DefaultEvaluator(BaseEvaluator):
    def __init__(
        self,
        task: BaseTask,
        backend: BaseBackend,
        run_metadata: dict[str, Any],
        postprocessing_config: PostprocessingConfig | None = None,
        no_image: bool = False,
        no_question: bool = False,
    ):
        super().__init__(task, backend)
        self.run_metadata = run_metadata
        self.postprocessing_config = postprocessing_config
        self.no_image = no_image
        self.no_question = no_question

    def evaluate(self, dataset: Iterable[dict[str, Any]]) -> pd.DataFrame:
        records = []
        for row in dataset:
            row = dict(row)
            prompt = self.task.build_prompt(row, no_question=self.no_question)
            image = row.pop("image")
            records.append(
                {
                    "image": None if self.no_image else image,
                    "prompt": prompt,
                    "row": row,
                }
            )

        generations = self.backend.generate_batch(
            [
                {
                    "image": record["image"],
                    "prompt": record["prompt"],
                    "task": self.task.name,
                    "row": record["row"],
                }
                for record in tqdm(records, desc=f"Preparing {self.task.name}")
            ]
        )

        rows = []
        for record, (raw_prediction, error) in tqdm(
            zip(records, generations, strict=True),
            total=len(records),
            desc=f"Scoring {self.task.name}",
        ):
            row, prompt = record["row"], record["prompt"]
            processed_prediction = postprocess(raw_prediction, self.postprocessing_config)
            score = self.task.score_prediction(processed_prediction, row)

            rows.append(
                {
                    "run_id": self.run_metadata["run_id"],
                    "model_id": self.run_metadata["model_id"],
                    "dataset_id": self.run_metadata["dataset_id"],
                    "dataset_revision": self.run_metadata["dataset_revision"],
                    "backend": self.run_metadata["backend"],
                    "task": self.task.name,
                    **row,
                    "score": score.score,
                    "scoring_method": score.scoring_method,
                    "predictions": [
                        {
                            "prompt": prompt,
                            "raw_prediction": raw_prediction,
                            "processed_prediction": processed_prediction,
                            "error": error,
                            **asdict(score),
                        }
                    ],
                }
            )

        return pd.DataFrame(rows)
