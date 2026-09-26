"""IntentForge pipeline: train -> evaluate -> benchmark -> predict.

Single entry point used by the CLI and the demo. Owns no ML logic itself; it
wires data, backends, router, HPO and evaluation together and records evidence.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from intentforge.core.config import Config
from intentforge.core.errors import PipelineError
from intentforge.core.types import BenchmarkReport, Prediction, RouterDecision
from intentforge.data.loaders import get_loader
from intentforge.domain.backends import XGBOOST_AVAILABLE, TRANSFORMER_AVAILABLE, build_backends
from intentforge.domain.cascade import cascade_predict
from intentforge.domain.router import C3RCascadeRouter
from intentforge.eval.metrics import classification_metrics, latency_profile, memory_rss_mb
from intentforge.hpo.tune import tune_router


def _versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for mod, attr in [
        ("sklearn", "__version__"),
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("xgboost", "__version__"),
        ("optuna", "__version__"),
        ("transformers", "__version__"),
    ]:
        try:
            m = __import__(mod)
            out[mod] = getattr(m, attr, "unknown")
        except Exception:
            out[mod] = None
    return out


class IntentForgePipeline:
    """End-to-end orchestrator."""

    def __init__(self, cfg: Config, data_source: str = "builtin", **loader_kwargs):
        self.cfg = cfg
        self.data_source = data_source
        self.loader_kwargs = loader_kwargs
        self.data: Dict[str, Any] = {}
        self.backends: Dict[str, Any] = {}
        self.router: Optional[C3RCascadeRouter] = None
        self.eval_result: Dict[str, Any] = {}
        self.hpo_result: Dict[str, Any] = {}
        # NOTE: named *_report deliberately — `benchmark` is a method, and an
        # instance attribute with the same name would shadow it.
        self.benchmark_report: Optional[BenchmarkReport] = None

    # --- train --------------------------------------------------------------
    def train(self) -> None:
        try:
            loader = get_loader(self.cfg, self.data_source, **self.loader_kwargs)
            self.data = loader.load()
        except Exception as exc:
            raise PipelineError(f"data load failed: {exc}", code="E500", cause=exc) from exc

        self.backends = build_backends(self.cfg)
        if not self.backends:
            raise PipelineError("no backend available", code="E500")

        for name, b in self.backends.items():
            b.fit(self.data["X_train"], self.data["y_train"])

        names = list(self.backends.keys())
        fast = names[0]
        strong = names[-1] if len(names) > 1 else names[0]
        self.router = C3RCascadeRouter(
            fast_backend=fast, strong_backend=strong,
            tau_high=self.cfg.tau_high, tau_low=self.cfg.tau_low,
        )
        self.router.fit(self.backends, self.data["X_val"], self.data["y_val"])

        if self.cfg.hpo_enabled and len(names) > 1:
            self.hpo_result = tune_router(
                self.router, self.backends, self.data["X_val"], self.data["y_val"],
                n_trials=self.cfg.hpo_trials,
            )
        else:
            self.hpo_result = {"tau_high": self.router.tau_high, "tau_low": self.router.tau_low}

    # --- evaluate -----------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        if not self.backends or self.router is None:
            raise PipelineError("call train() before evaluate()", code="E500")
        y_test = self.data["y_test"]
        per_backend: Dict[str, Dict[str, float]] = {}
        for name, b in self.backends.items():
            preds = b.predict(self.data["X_test"])
            per_backend[name] = classification_metrics(y_test, [p.label for p in preds])

        cpreds, decisions = cascade_predict(self.router, self.backends, self.data["X_test"])
        cm = classification_metrics(y_test, [p.label for p in cpreds])
        n = max(1, len(decisions))
        frac_fast = sum(1 for d in decisions if d.chosen_backend == self.router.fast_name and not d.abstain) / n
        frac_strong = sum(1 for d in decisions if d.chosen_backend == self.router.strong_name and not d.abstain) / n
        frac_abstain = sum(1 for d in decisions if d.abstain) / n

        self.eval_result = {
            "per_backend": per_backend,
            "cascade": {
                **cm,
                "frac_fast": round(frac_fast, 4),
                "frac_strong": round(frac_strong, 4),
                "frac_abstain": round(frac_abstain, 4),
                "tau_high": round(self.router.tau_high, 4),
                "tau_low": round(self.router.tau_low, 4),
            },
        }
        return self.eval_result

    # --- benchmark (real execution + evidence) -------------------------------
    # --- lazy (true) cascade ------------------------------------------------
    def _cascade_one(self, index: int, text: str):
        """Run the real cascade for a single request.

        The strong backend is invoked **only** when the fast one is not
        confident enough — this is where the latency saving comes from.
        """
        if self.router is None or not self.backends:
            raise PipelineError("call train() first", code="E500")
        router = self.router
        fast_pred = self.backends[router.fast_name].predict([text])[0]
        confident, cal = router.stage1(fast_pred)
        if confident:
            return fast_pred, RouterDecision(index, router.fast_name, cal, False, "fast_confident")

        has_strong = (
            router.strong_name in self.backends
            and router.strong_name != router.fast_name
        )
        if has_strong:
            strong_pred = self.backends[router.strong_name].predict([text])[0]
            decision = router.stage2(index, fast_pred, strong_pred)
            chosen = (
                strong_pred if decision.chosen_backend == router.strong_name else fast_pred
            )
            return chosen, decision

        # single-backend fallback: no escalation possible -> abstain when unsure
        return fast_pred, RouterDecision(
            index, router.fast_name, cal, True, "abstain_low_conf"
        )

    def _cascade_one_batch(self, texts: List[str]) -> List[Prediction]:
        return [self._cascade_one(i, t)[0] for i, t in enumerate(texts)]

    def benchmark(self) -> BenchmarkReport:
        if not self.backends or self.router is None:
            raise PipelineError("call train() before benchmark()", code="E500")
        sample = self.data["X_test"][: self.cfg.benchmark_sample]

        mem_before = memory_rss_mb()

        # Fair comparison: every row below measures *per-request* latency
        # (batch-of-1 through the same code path), because that is what the
        # cascade actually does. Batch throughput is recorded as a secondary
        # metric — it is much faster but not comparable to the cascade number.
        per_backend_lat: Dict[str, Dict[str, float]] = {}
        for name, b in self.backends.items():
            def per_request(texts, _b=b):
                return [_b.predict([t])[0] for t in texts]

            req = latency_profile(per_request, sample, runs=self.cfg.latency_runs)
            batch = latency_profile(b.predict, sample, runs=self.cfg.latency_runs)
            per_backend_lat[name] = {
                **req,
                "batch_mean_ms": batch["mean_ms"],
                "batch_throughput_per_s": batch["throughput_per_s"],
            }

        cascade_lat = latency_profile(self._cascade_one_batch, sample, runs=self.cfg.latency_runs)
        mem_after = memory_rss_mb()

        env = {
            "os": f"{platform.system()} {platform.release()}",
            "python": sys.version.split()[0],
            "cpu_count": _cpu_count(),
            "xgboost_available": XGBOOST_AVAILABLE,
            "transformer_available": TRANSFORMER_AVAILABLE,
            "versions": _versions(),
            "memory_rss_mb_start": mem_before,
            "memory_rss_mb_end": mem_after,
            "memory_rss_mb": mem_after,
        }
        dataset = {
            "source": self.data_source,
            "n_train": len(self.data["X_train"]),
            "n_test": len(self.data["X_test"]),
            "n_classes": len(self.data["labels"]),
            "benchmark_sample": len(sample),
        }

        self.benchmark_report = BenchmarkReport(
            environment=env,
            dataset=dataset,
            per_backend=per_backend_lat,
            cascade={
                **cascade_lat,
                "macro_f1": self.eval_result.get("cascade", {}).get("macro_f1", 0.0),
                "accuracy": self.eval_result.get("cascade", {}).get("accuracy", 0.0),
                "frac_fast": self.eval_result.get("cascade", {}).get("frac_fast", 0.0),
                "frac_strong": self.eval_result.get("cascade", {}).get("frac_strong", 0.0),
                "frac_abstain": self.eval_result.get("cascade", {}).get("frac_abstain", 0.0),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.cfg.to_dict(),
        )
        return self.benchmark_report

    # --- live predict -------------------------------------------------------
    def predict(self, texts: List[str]):
        if not self.backends or self.router is None:
            raise PipelineError("call train() before predict()", code="E500")
        out = []
        for i, t in enumerate(texts):
            chosen, d = self._cascade_one(i, t)
            out.append(
                {
                    "text": t,
                    "label": chosen.label,
                    "backend": d.chosen_backend,
                    "confidence": round(chosen.confidence, 4),
                    "calibrated_confidence": round(d.confidence, 4),
                    "abstain": d.abstain,
                    "reason": d.reason,
                }
            )
        return out

    # --- reporting ----------------------------------------------------------
    def print_report(self) -> None:
        ev = self.eval_result or self.evaluate()
        print("\n=== IntentForge · Evaluation ===")
        hdr = f"{'backend':<16}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>12}"
        print(hdr)
        print("-" * len(hdr))
        for name, m in ev["per_backend"].items():
            print(f"{name:<16}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['weighted_f1']:>12.4f}")
        c = ev["cascade"]
        print("-" * len(hdr))
        print(f"{'C3R cascade':<16}{c['accuracy']:>10.4f}{c['macro_f1']:>10.4f}{c['weighted_f1']:>12.4f}")
        print(
            f"  routing -> fast:{c['frac_fast']:.2%}  strong:{c['frac_strong']:.2%}  "
            f"abstain:{c['frac_abstain']:.2%}  (tau_h={c['tau_high']}, tau_l={c['tau_low']})"
        )
        if self.benchmark_report:
            print("\n=== Latency baseline (CPU) ===")
            lh = f"{'backend':<16}{'p50_ms':>10}{'p95_ms':>10}{'mean_ms':>10}{'thru/s':>10}"
            print(lh)
            print("-" * len(lh))
            for name, m in self.benchmark_report.per_backend.items():
                print(f"{name:<16}{m['p50_ms']:>10}{m['p95_ms']:>10}{m['mean_ms']:>10}{m['throughput_per_s']:>10}")
            cb = self.benchmark_report.cascade
            print(f"{'C3R cascade':<16}{cb['p50_ms']:>10}{cb['p95_ms']:>10}{cb['mean_ms']:>10}{cb['throughput_per_s']:>10}")

    def save_benchmark(self, path: str) -> None:
        if self.benchmark_report is None:
            raise PipelineError("run benchmark() first", code="E500")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.benchmark_report.__dict__, fh, indent=2, default=str)
        print(f"[intentforge] benchmark written -> {path}")


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0
