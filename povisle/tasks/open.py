from typing import Any

from povisle.tasks.base import BaseTask, ParsingMethod, ParsingStatus, ScoringMethod, TaskScore
from povisle.utils import normalize_text


class OpenEndedTask(BaseTask):
    name = "open"

    def build_prompt(self, row: dict[str, Any], no_question: bool = False) -> str:
        if no_question:
            return ""
        return row["question"].strip()

    def score_prediction(self, raw_prediction: str | None, row: dict[str, Any]) -> TaskScore:
        parsed = str(raw_prediction or "").strip()
        if not parsed:
            return TaskScore(
                parsed_prediction=None,
                parsing_status=ParsingStatus.UNPARSED,
                parsing_method=ParsingMethod.EMPTY,
                score=0.0,
                scoring_method=ScoringMethod.NORMALIZED_EXACT,
            )

        gold_answer = row["answer"].strip()
        normalized_prediction = normalize_text(parsed)
        normalized_gold = normalize_text(gold_answer)

        is_exact = normalized_prediction == normalized_gold
        is_substring = bool(normalized_gold) and normalized_gold in normalized_prediction
        scoring_method = ScoringMethod.NORMALIZED_EXACT if is_exact else ScoringMethod.NORMALIZED_SUBSTRING
        
        return TaskScore(
            parsed_prediction=parsed,
            parsing_status=ParsingStatus.PARSED,
            parsing_method=ParsingMethod.RAW_TEXT,
            score=1.0 if is_exact or is_substring else 0.0,
            scoring_method=scoring_method,
        )

    def gold_answer(self, row: dict[str, Any]) -> str:
        return str(row.get("answer", "")).strip()
