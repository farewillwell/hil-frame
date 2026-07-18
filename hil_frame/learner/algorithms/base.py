from __future__ import annotations

from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def update(self, batch: dict) -> dict:
        ...

