import re
from typing import Any

from povisle.tasks.base import BaseTask, ParsingMethod, ParsingStatus, ScoringMethod, TaskScore
from povisle.utils import normalize_text


YES_MARKERS = {"tak", "yes", "true", "prawda"}
NO_MARKERS = {"nie", "no", "false", "falsz", "fasz"}
LABELS = ("tak", "nie")


class YesNoTask(BaseTask):
    name = "yn"

    def build_prompt(self, row: dict[str, Any], no_question: bool = False) -> str:
        if no_question:
            return "Odpowiedz tylko TAK lub NIE."
        return str(row.get("question", "")).strip()

    def score_prediction(self, raw_prediction: str | None, row: dict[str, Any]) -> TaskScore:
        parsed, status, method = parse_yes_no_prediction(raw_prediction)
        gold = self.gold_answer(row)
        is_correct = parsed == gold if parsed else False
        return TaskScore(
            parsed_prediction=parsed,
            parsing_status=status,
            parsing_method=method,
            score=1.0 if is_correct else 0.0,
            scoring_method=ScoringMethod.EXACT_LABEL,
        )

    def gold_answer(self, row: dict[str, Any]) -> str:
        return canonical_yes_no(row.get("answer"))


def canonical_yes_no(value: Any) -> str:
    normalized = normalize_text(str(value or ""))
    if normalized in YES_MARKERS:
        return "tak"
    if normalized in NO_MARKERS:
        return "nie"
    return normalized


def parse_yes_no_prediction(raw_prediction: Any) -> tuple[str | None, ParsingStatus, ParsingMethod]:
    text = str(raw_prediction or "").strip()
    if not text:
        return None, ParsingStatus.UNPARSED, ParsingMethod.EMPTY

    normalized = normalize_text(text)
    if normalized in YES_MARKERS:
        return "tak", ParsingStatus.PARSED, ParsingMethod.EXACT_YES_NO
    if normalized in NO_MARKERS:
        return "nie", ParsingStatus.PARSED, ParsingMethod.EXACT_YES_NO

    for marker in YES_MARKERS:
        if normalized.startswith(marker):
            return "tak", ParsingStatus.PARSED, ParsingMethod.SENTENCE_PREFIX
    for marker in NO_MARKERS:
        if normalized.startswith(marker):
            return "nie", ParsingStatus.PARSED, ParsingMethod.SENTENCE_PREFIX

    match = re.search(r"\b(tak|yes|true|prawda|nie|no|false|falsz|fasz)\b", normalized)
    if match:
        token = match.group(1)
        status = ParsingStatus.PARSED
        method = ParsingMethod.FIRST_TOKEN
        return ("tak", status, method) if token in YES_MARKERS else ("nie", status, method)

    return None, ParsingStatus.UNPARSED, ParsingMethod.NO_YES_NO_MATCH
