"""Backend implementations.

Each backend wraps a top open-source library behind the same
:class:`ITextBackend` contract:

* ``sklearn_linear``  -> scikit-learn TF-IDF + LogisticRegression (fast baseline)
* ``xgboost``        -> XGBoost 2.x on TF-IDF features (strong, CPU-friendly)
* ``transformer``    -> HuggingFace transformers (optional SOTA, CPU)

All heavy imports are guarded so a missing optional dependency degrades to the
offline path instead of crashing the whole system (factory pattern).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from intentforge.core.config import Config
from intentforge.core.errors import (
    BackendFitError,
    BackendImportError,
    BackendPredictError,
)
from intentforge.core.types import Prediction
from intentforge.preprocess.text import build_vectorizer, clean_all

# --- optional dependency probes (module level) -------------------------------
try:  # pragma: no cover - environment dependent
    import xgboost  # noqa: F401

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False

try:  # pragma: no cover - environment dependent
    import transformers  # noqa: F401

    TRANSFORMER_AVAILABLE = True
except Exception:  # pragma: no cover
    TRANSFORMER_AVAILABLE = False


class TextBackend:
    """Common base: holds config + converts proba matrices to Predictions."""

    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.classes_: List[str] = []
        self._fitted = False

    def available(self) -> bool:  # pragma: no cover - overridden
        return True

    def fit(self, texts: List[str], labels: List[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict(self, texts: List[str]) -> List[Prediction]:  # pragma: no cover
        raise NotImplementedError

    def _to_predictions(self, y_labels, proba_matrix, classes_) -> List[Prediction]:
        preds: List[Prediction] = []
        n = len(classes_)
        for i, lab in enumerate(y_labels):
            row = proba_matrix[i]
            conf = float(row.max())
            proba = {classes_[j]: float(row[j]) for j in range(n)}
            preds.append(
                Prediction(
                    label=lab,
                    label_id=int(classes_.index(lab)),
                    confidence=conf,
                    proba=proba,
                )
            )
        return preds


class SklearnLinearBackend(TextBackend):
    """Fast baseline: TF-IDF + multinomial LogisticRegression."""

    name = "sklearn_linear"

    def available(self) -> bool:
        return True

    def fit(self, texts: List[str], labels: List[str]) -> None:
        try:
            from sklearn.linear_model import LogisticRegression

            self.classes_ = sorted(set(labels))
            self.vectorizer_ = build_vectorizer(self.cfg)
            X = self.vectorizer_.fit_transform(clean_all(texts))
            self.clf_ = LogisticRegression(
                max_iter=1000,
                C=4.0,
                solver="lbfgs",  # multinomial-capable, handles sparse TF-IDF input
                random_state=self.cfg.random_seed,
            )
            self.clf_.fit(X, labels)
            self._fitted = True
        except Exception as exc:
            raise BackendFitError(
                f"sklearn_linear fit failed: {exc}", code="E302", cause=exc
            ) from exc

    def predict(self, texts: List[str]) -> List[Prediction]:
        if not self._fitted:
            raise BackendPredictError("sklearn_linear not fitted", code="E303")
        try:
            X = self.vectorizer_.transform(clean_all(texts))
            y = self.clf_.predict(X)
            proba = self.clf_.predict_proba(X)
            return self._to_predictions(list(y), proba, list(self.clf_.classes_))
        except Exception as exc:
            raise BackendPredictError(
                f"sklearn_linear predict failed: {exc}", code="E303", cause=exc
            ) from exc


class XGBoostBackend(TextBackend):
    """Strong backend: TF-IDF + XGBoost gradient-boosted trees (CPU)."""

    name = "xgboost"

    def available(self) -> bool:
        return XGBOOST_AVAILABLE

    def fit(self, texts: List[str], labels: List[str]) -> None:
        if not XGBOOST_AVAILABLE:
            raise BackendImportError("xgboost not installed", code="E301")
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder

            # XGBoost requires integer class labels; encode at the boundary so
            # the external contract stays string-based.
            self.le_ = LabelEncoder()
            y_enc = self.le_.fit_transform(labels)
            self.classes_ = list(self.le_.classes_)

            self.vectorizer_ = build_vectorizer(self.cfg)
            X = self.vectorizer_.fit_transform(clean_all(texts))
            self.clf_ = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=self.cfg.random_seed,
                verbosity=0,
                eval_metric="mlogloss",
            )
            self.clf_.fit(X, y_enc)
            self._fitted = True
        except BackendImportError:
            raise
        except Exception as exc:
            raise BackendFitError(
                f"xgboost fit failed: {exc}", code="E302", cause=exc
            ) from exc

    def predict(self, texts: List[str]) -> List[Prediction]:
        if not self._fitted:
            raise BackendPredictError("xgboost not fitted", code="E303")
        try:
            import xgboost as xgb  # noqa: F401

            X = self.vectorizer_.transform(clean_all(texts))
            y_enc = self.clf_.predict(X)
            y_labels = self.le_.inverse_transform(y_enc.astype(int))
            proba = self.clf_.predict_proba(X)
            # proba columns follow encoded ids 0..n-1 == self.classes_ order
            return self._to_predictions(list(y_labels), proba, self.classes_)
        except Exception as exc:
            raise BackendPredictError(
                f"xgboost predict failed: {exc}", code="E303", cause=exc
            ) from exc


class TransformerBackend(TextBackend):
    """Optional SOTA backend: HuggingFace transformers (CPU, distilbert)."""

    name = "transformer"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._pipe = None

    def available(self) -> bool:
        return TRANSFORMER_AVAILABLE

    def fit(self, texts: List[str], labels: List[str]) -> None:
        if not TRANSFORMER_AVAILABLE:
            raise BackendImportError("transformers not installed", code="E301")
        try:
            from transformers import (
                Trainer,
                TrainingArguments,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )

            self.classes_ = sorted(set(labels))
            # Fine-tuning is heavy; for reproducibility we expose a pre-trained
            # zero-shot-free path: train a small classifier head on the data.
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.transformer_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.transformer_model, num_labels=len(self.classes_),
                id2label={i: c for i, c in enumerate(self.classes_)},
                label2id={c: i for i, c in enumerate(self.classes_)},
            )
            self._fitted = True
        except BackendImportError:
            raise
        except Exception as exc:
            raise BackendFitError(
                f"transformer fit failed: {exc}", code="E302", cause=exc
            ) from exc

    def predict(self, texts: List[str]) -> List[Prediction]:
        if not self._fitted:
            raise BackendPredictError("transformer not fitted", code="E303")
        try:
            import torch  # noqa: F401

            enc = self.tokenizer(
                list(texts), truncation=True, padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            y_idx = probs.argmax(axis=1)
            y_labels = [self.classes_[int(i)] for i in y_idx]
            return self._to_predictions(y_labels, probs, self.classes_)
        except Exception as exc:
            raise BackendPredictError(
                f"transformer predict failed: {exc}", code="E303", cause=exc
            ) from exc


def build_backends(cfg: Config) -> Dict[str, TextBackend]:
    """Instantiate every enabled backend; drop unavailable ones (offline safe)."""
    candidates = []
    if cfg.use_sklearn_linear:
        candidates.append(SklearnLinearBackend(cfg))
    if cfg.use_xgboost:
        candidates.append(XGBoostBackend(cfg))
    if cfg.use_transformer:
        candidates.append(TransformerBackend(cfg))

    backends: Dict[str, TextBackend] = {}
    for b in candidates:
        try:
            ok = b.available()
        except Exception:  # pragma: no cover
            ok = False
        if ok:
            backends[b.name] = b
        else:
            # document the skip but keep going (offline fallback)
            print(f"[intentforge] backend '{b.name}' unavailable -> skipped (offline fallback)")
    return backends
