from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_batch(self, obs: Sequence[Dict]) -> list[Dict]:
        """Infer actions for a batch of observations.

        Policies should override this method to perform a single batched model call.
        The fallback keeps existing third-party policies compatible.
        """
        return [self.infer(element) for element in obs]

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
