from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParsingStatus(str, Enum):
    PARSED = "parsed"
    UNPARSED = "unparsed"


class ParsingMethod(str, Enum):
    EMPTY = "empty"
    EMPTY_PREDICTION = "empty_prediction"
    EXACT_LABEL = "exact_label"
    PREFIXED_LABEL = "prefixed_label"
    LEADING_LABEL = "leading_label"
    STANDALONE_LABEL = "standalone_label"
    CHOICE_TEXT_EXACT = "choice_text_exact"
    CHOICE_TEXT_MATCH = "choice_text_match"
    NO_LABEL_MATCH = "no_label_match"
    EXACT_YES_NO = "exact_yes_no"
    SENTENCE_PREFIX = "sentence_prefix"
    FIRST_TOKEN = "first_token"
    NO_YES_NO_MATCH = "no_yes_no_match"
    RAW_TEXT = "raw_text"


class ScoringMethod(str, Enum):
    EXACT_LABEL = "exact_label"
    NORMALIZED_EXACT = "normalized_exact"
    NORMALIZED_SUBSTRING = "normalized_substring"
    CIRCULAR = "circular"


@dataclass(frozen=True)
class TaskScore:
    parsed_prediction: str | None
    parsing_status: ParsingStatus
    parsing_method: ParsingMethod
    score: float
    scoring_method: ScoringMethod


class BaseTask:
    name: str

    def build_prompt(self, row: dict[str, Any], no_question: bool = False) -> str:
        raise NotImplementedError

    def score_prediction(self, raw_prediction: str | None, row: dict[str, Any]) -> TaskScore:
        raise NotImplementedError

    @staticmethod
    def from_name(name: str) -> "BaseTask":
        if name == "mcq":
            from povisle.tasks.mcq import MultipleChoiceTask

            return MultipleChoiceTask()
        elif name == "yn":
            from povisle.tasks.yn import YesNoTask

            return YesNoTask()
        elif name == "open":
            from povisle.tasks.open import OpenEndedTask

            return OpenEndedTask()
        else:
            raise ValueError(f"Unknown task name: {name}")
