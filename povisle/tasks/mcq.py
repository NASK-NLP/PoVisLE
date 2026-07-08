from __future__ import annotations

import re
from typing import Any

from povisle.tasks.base import BaseTask, ParsingMethod, ParsingStatus, ScoringMethod, TaskScore
from povisle.utils import normalize_text


LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


def choice_map(row: dict[str, Any]) -> dict[str, str]:
    return {label: row[label].strip() for label in LABELS if isinstance(row.get(label), str) and row[label].strip()}


def parse_prediction(raw_prediction: str | None, choices: dict[str, str]) -> tuple[str | None, ParsingStatus, ParsingMethod]:
    text = str(raw_prediction or "").strip()
    if not text:
        return None, ParsingStatus.UNPARSED, ParsingMethod.EMPTY_PREDICTION

    upper = text.upper()
    exact = re.fullmatch(r"[\(\[]?([A-H])[\)\].:\-]?", upper)
    if exact:
        return exact.group(1), ParsingStatus.PARSED, ParsingMethod.EXACT_LABEL

    prefixed = re.search(
        r"(?i)^\s*(?:odpowiedz|answer|ans|option|wybieram|poprawna odpowiedz(?: to)?)\s*[:\-]?\s*[\(\[]?([A-H])",
        text,
    )
    if prefixed:
        return prefixed.group(1).upper(), ParsingStatus.PARSED, ParsingMethod.PREFIXED_LABEL

    leading = re.search(r"^\s*[\(\[]?([A-Ha-h])[\)\].:\-]\s+", text)
    if leading:
        return leading.group(1).upper(), ParsingStatus.PARSED, ParsingMethod.LEADING_LABEL

    standalone = sorted(set(re.findall(r"\b([A-H])\b", upper)))
    if len(standalone) == 1:
        return standalone[0], ParsingStatus.PARSED, ParsingMethod.STANDALONE_LABEL

    normalized = normalize_text(text)
    exact_text_matches = [label for label, option in choices.items() if normalize_text(option) == normalized]
    if len(exact_text_matches) == 1:
        return exact_text_matches[0], ParsingStatus.PARSED, ParsingMethod.CHOICE_TEXT_EXACT

    contained_matches = [label for label, option in choices.items() if normalize_text(option) and normalize_text(option) in normalized]
    if len(contained_matches) == 1:
        return contained_matches[0], ParsingStatus.PARSED, ParsingMethod.CHOICE_TEXT_MATCH

    return None, ParsingStatus.UNPARSED, ParsingMethod.NO_LABEL_MATCH


class MultipleChoiceTask(BaseTask):
    name = "mcq"

    def build_prompt(self, row: dict[str, Any], no_question: bool = False) -> str:
        lines = []
        if no_question:
            lines.append("Odpowiedz tylko poprawną literką.")
        else:
            lines.append(row["question"].strip())
        lines.extend(f"{label}. {value.strip()}" for label, value in choice_map(row).items())
        return "\n".join(lines).strip()

    def score_prediction(self, raw_prediction: str | None, row: dict[str, Any]) -> TaskScore:
        parsed, status, method = parse_prediction(raw_prediction, choice_map(row))
        if not parsed:
            return TaskScore(
                parsed_prediction=None,
                parsing_status=status,
                parsing_method=method,
                score=0.0,
                scoring_method=ScoringMethod.EXACT_LABEL,
            )

        gold_answer = row["answer"].strip().upper()
        is_correct = parsed == gold_answer

        return TaskScore(
            parsed_prediction=parsed,
            parsing_status=status,
            parsing_method=method,
            score=1.0 if is_correct else 0.0,
            scoring_method=ScoringMethod.EXACT_LABEL,
        )
