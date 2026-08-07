import unicodedata
from collections.abc import Iterable
from math import isnan
from typing import Any

from povisle.tasks.base import BaseTask, ParsingMethod, ParsingStatus, ScoringMethod, TaskScore


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

        answers = accepted_open_answers(row)
        check_casing = row.get("check_casing", False)
        check_diacritics = row.get("check_diacritics", True)

        is_exact = any(
            normalize_open_answer(parsed, answer, check_casing=check_casing, check_diacritics=check_diacritics)
            == normalize_open_answer(answer, answer, check_casing=check_casing, check_diacritics=check_diacritics)
            for answer in answers
        )

        return TaskScore(
            parsed_prediction=parsed,
            parsing_status=ParsingStatus.PARSED,
            parsing_method=ParsingMethod.RAW_TEXT,
            score=1.0 if is_exact else 0.0,
            scoring_method=ScoringMethod.NORMALIZED_EXACT,
        )

    def gold_answer(self, row: dict[str, Any]) -> str:
        return str(row.get("answer", "")).strip()


def accepted_open_answers(row: dict[str, Any]) -> list[str]:
    answers = [str(row.get("answer", "")).strip()]
    includes = row.get("include")

    if is_missing_value(includes):
        includes = []
    elif isinstance(includes, str):
        includes = [includes]
    elif not isinstance(includes, Iterable):
        includes = []

    answers.extend(str(answer).strip() for answer in includes if str(answer).strip())
    return [answer for answer in answers if answer]


def is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and isnan(value)) or type(value).__name__ == "NAType"


def normalize_open_answer(
    text: str,
    gold_answer: str,
    *,
    check_casing: bool = False,
    check_diacritics: bool = True,
) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    if not check_diacritics:
        normalized = strip_diacritics(normalized)
    if not check_casing:
        normalized = normalized.casefold()
    normalized = normalized.strip()

    if not has_punctuation(gold_answer):
        normalized = strip_punctuation(normalized)

    return " ".join(normalized.split())


def has_punctuation(text: str) -> bool:
    return any(unicodedata.category(char).startswith("P") for char in str(text or ""))


def strip_punctuation(text: str) -> str:
    return "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.category(char).startswith("M"))
    return unicodedata.normalize("NFC", stripped)
