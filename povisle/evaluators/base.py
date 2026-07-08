from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import pandas as pd

from povisle.backends.base import BaseBackend
from povisle.tasks.base import BaseTask


class BaseEvaluator(ABC):
    def __init__(self, task: BaseTask, backend: BaseBackend):
        self.task = task
        self.backend = backend

    @abstractmethod
    def evaluate(self, dataset: Iterable[dict[str, Any]]) -> pd.DataFrame:
        raise NotImplementedError
