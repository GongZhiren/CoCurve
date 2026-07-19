"""CoCurve pruning package.

Logical structure:
- cocurve.core: pruning/math core
- cocurve.evaluation: eval/prompt/scoring
- cocurve.infra: config/io/types/utils
"""

from typing import Any

__all__ = ["CoCurvePipeline"]


def __getattr__(name: str) -> Any:
    # Keep lightweight imports (e.g. cocurve.config) free from heavy
    # dependencies such as transformers unless pipeline is explicitly requested.
    if name == "CoCurvePipeline":
        from .pipeline import CoCurvePipeline as _CoCurvePipeline

        return _CoCurvePipeline
    raise AttributeError(name)
