"""Shared cascade execution used by HPO and the pipeline.

Backends are predicted in batch once (efficient), then the router decides per
sample. This is the single source of truth for "what the system predicts".
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from intentforge.core.types import Prediction, RouterDecision
from intentforge.core.interfaces import ITextBackend


def cascade_predict(router, backends: Dict[str, ITextBackend], texts: List[str]):
    """Return ``(predictions, decisions)`` for a list of texts."""
    all_outputs: Dict[str, List[Prediction]] = {
        name: b.predict(texts) for name, b in backends.items()
    }
    preds: List[Prediction] = []
    decisions: List[RouterDecision] = []
    for i in range(len(texts)):
        outputs = {name: all_outputs[name][i] for name in backends}
        d = router.route(i, outputs)
        chosen = outputs.get(d.chosen_backend)
        if chosen is None and outputs:
            chosen = next(iter(outputs.values()))
        if chosen is None:  # pragma: no cover - only if no backends at all
            chosen = Prediction(label="<unk>", label_id=-1, confidence=0.0)
        preds.append(chosen)
        decisions.append(d)
    return preds, decisions
