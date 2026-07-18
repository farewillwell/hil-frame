from __future__ import annotations

from .transition_view import TransitionView


class TransitionSampler:
    def __init__(self, view: TransitionView) -> None:
        self.view = view

    def sample(self, batch_size: int, **filters):
        return self.view.sample_transitions(batch_size, **filters)

