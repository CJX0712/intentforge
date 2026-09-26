# IntentForge

**Confidence-Calibrated Cascaded Routing for text classification.**
A modular, CPU-only system that routes every request to the cheapest model that
will get it right — and abstains instead of guessing when even the strong model
is unsure.

> Author: **晨星** · License: Apache-2.0 · Built on scikit-learn, XGBoost and Optuna.

---

## TL;DR

On an off-the-shelf CPU laptop, IntentForge matches the **strong** model's
quality while running **2.27× faster**, by sending only the genuinely hard
requests down the expensive path.

| Path | Accuracy | Macro-F1 | Latency (mean, per request) | Throughput |
|---|---|---|---|---|
| `sklearn_linear` (fast) | 0.8183 | 0.8247 | 0.9466 ms | 1 056 /s |
| `xgboost` (strong) | 0.8267 | 0.8338 | 4.9875 ms | 200 /s |
| **C3R cascade** | **0.8267** | **0.8338** | **2.1978 ms** | **455 /s** |

Everything below (including those numbers) is reproducible in one command.

---

## 🧠 Innovation: C3R — Confidence-Calibrated Cascaded Router

Naive confidence scores are **not** probabilities. `max p(y|x)` from XGBoost and
from logistic regression live on different scales, so comparing either against a
fixed threshold is unsound — yet that is what every "if uncertain, escalate"
heuristic does.

**C3R fixes the calibration problem before it touches routing.**

1. **Calibrate.** For each backend *b*, fit a monotone map on a held-out set:
   `g_b = IsotonicRegression( c_b(x_i) → 1[pred_b(x_i) = y_i] )`.
   `g_b(c)` now means *"probability this backend is correct"*, comparable across
   backends and directly interpretable as a threshold.
2. **Route lazily.** Evaluate the *fast* backend first. Only if it is not
   confident enough is the *strong* backend ever invoked — so the expensive model
   is paid for on the hard samples only.
3. **Abstain.** When even the strong model is below `tau_low`, the request is
   flagged for a human/fallback rather than answered with a coin flip.
4. **Tune.** `tau_high` / `tau_low` are optimized by Optuna against macro-F1,
   with a penalty that keeps a healthy share on the fast path.

```
input ──► fast backend ──► g_fast(c) ≥ tau_high ? ──yes──► answer (fast only)
                                │ no
                                ▼
                        strong backend ──► g_strong(c) ≥ tau_low ? ──yes──► escalate
                                │ no
                                ▼
                            ABSTAIN (flag, don't guess)
```

**Two corollaries worth calling out:**

* *Decision equivalence.* The eager batch path (`route()`) and the lazy path
  (`stage1()` + `stage2()`) produce **identical decisions**, so accuracy can be
  measured cheaply in batch while latency is measured honestly per request. This
  is asserted in the test suite.
* *Calibrated abstention is learnable.* `tau_low` is a business knob: lower it to
  automate more, raise it to protect precision. It is measured, not guessed.

---

## ✨ Features

- ✅ **Modular & contract-first** — every component behind a `Protocol` with typed I/O, protocol semantics and `E1xx–E5xx` error codes
- ✅ **Top OSS, not bespoke ML** — scikit-learn, XGBoost, Optuna do the heavy lifting; optional HuggingFace path
- ✅ **Real offline fallback** — heavy/absent dependencies degrade gracefully instead of crashing
- ✅ **Honest benchmarking** — filters wet fingers: per-request latency compared apples-to-apples, evidenced in `benchmark.json`
- ✅ **Reproducible** — seeded synthetic corpus + `requirements.lock.txt` (full transitive lock)
- ✅ **Independently verifiable** — unit tests per module, minimal CLI examples, integration test asserting the innovation claim

---

## 🚀 Quickstart

### One-click (cross-platform)
```bash
python scripts/bootstrap.py      # creates .venv, installs lock, runs tests, runs demo
```

### Manual
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt   # Windows
# .venv/bin/python -m pip install -r requirements.lock.txt         # Linux/macOS

python -m pytest tests -q -W ignore::UserWarning     # 30 tests
python intentforge/examples/run_demo.py              # train + evaluate + benchmark
```

### Docker
```bash
docker build -t intentforge .
docker run --rm intentforge
```

### CLI
```bash
python -m intentforge.cli demo                       # everything
python -m intentforge.cli train
python -m intentforge.cli evaluate
python -m intentforge.cli benchmark --out benchmark.json
python -m intentforge.cli predict --text "check my balance please" "i want to cancel and talk to a human"
```

Every setting is overridable via ENV, e.g. `INTENTFORGE_N_SAMPLES=5000`,
`INTENTFORGE_USE_TRANSFORMER=true`, `INTENTFORGE_RANDOM_SEED=7`.

---

## 🏗️ Architecture

Unidirectional, acyclic. Nothing learns on its own; the pipeline wires everything.

```
cli.py / examples/run_demo.py
        │
        ▼
pipeline/pipeline.py ──► data/{loaders,generator}.py
        │            ├──► domain/{backends,router,cascade}.py
        │            ├──► hpo/tune.py
        │            ├──► eval/metrics.py
        │            └──► preprocess/text.py
        │
        └──► core/{types,errors,config,interfaces}.py     (dependency-free base)
```

| Module | Responsibility | Key interface | Error codes |
|---|---|---|---|
| `core` | types / errors / config / protocols | `ITextBackend`, `IRouter`, `IDataLoader`, `IEvaluator` | `E100` |
| `data` | seeded synthetic corpus + loaders | `load() -> splits` | `E200` |
| `preprocess` | cleaning + shared TF-IDF factory | `build_vectorizer(cfg)` | `E200` |
| `domain/backends` | one adapter per OSS library | `fit / predict / available` | `E301 E302 E303` |
| `domain/router` | **C3R** calibration + policy | `stage1 / stage2 / route` | `E401 E402` |
| `domain/cascade` | batch cascade (accuracy path) | `cascade_predict(...)` | — |
| `hpo` | Optuna threshold search | `tune_router(...)` | `E100` |
| `eval` | quality + latency + memory | `classification_metrics / latency_profile` | `E500` |
| `pipeline` | orchestration + reporting | `train / evaluate / benchmark / predict` | `E500` |

Full design notes, contracts and the reasoning behind each number: [`docs/architecture.md`](docs/architecture.md).

---

## 📦 Technology selection

Chosen by comparing performance, ecosystem, license and **recent activity**.

| Role | Candidates considered | Selected | License | Why |
|---|---|---|---|---|
| Fast backend | LogisticRegression · LinearSVC · ComplementNB | **LogisticRegression (TF-IDF)** | BSD-3 | Multinomial, sparse-friendly, calibrated-ish probabilities needed by C3R |
| Strong backend | XGBoost 2.x/3.x · LightGBM · CatBoost | **XGBoost 3.4.1** | Apache-2.0 | Best-in-class GBDT, mature Windows CPU wheels, sklearn-native API |
| HPO | Optuna · Ray Tune · GridSearchCV | **Optuna 5.0.0** | MIT | Define-by-run, pruning, extremely active (recent releases in 2026) |
| Metrics / vectorization | scikit-learn | **scikit-learn 1.9.1** | BSD-3 | Standard metrics; active (1.6+ with free-threaded 3.13 wheels) |
| Optional SOTA | HF transformers (distilbert) | **transformers** (optional) | Apache-2.0 | ⚪ Not exercised here (no torch installed) — see Offline fallback |

**Nothing was re-implemented from scratch.** The one piece of genuine new work is
the router policy itself (C3R), which is exactly where the reuse argument stops
applying — no existing library provides per-sample calibrated cascading with
abstention across heterogeneous backends.

### Offline fallback matrix

| Missing | System behaviour |
|---|---|
| `xgboost` | fast-only; cascade degrades to pass-through + abstain. Still trains and serves. |
| `transformers`/`torch` | optional backend skipped; classical path fully offline. |
| `psutil` | memory reported as `0.0`; latency/quality unaffected. |
| `optuna` | set `hpo_enabled=False` to use fixed thresholds. |

---

## 📊 Performance baseline

**Environment** — Windows 11 · Python 3.13.14 · **CPU-only (no CUDA)** · AMD Ryzen (16 logical cores) · 15.26 GB RAM · sklearn 1.9.1 · xgboost 3.4.1 · optuna 5.0.0 · numpy 2.5.3 · psutil 7.2.2.
**Dataset** — builtin synthetic intent corpus, 10 intents, 1 530 train / 600 test, seed 42.
**Method** — per-request latency (batch-of-1), 600 samples × 3 repetitions, fixed seeds.

| Metric | `sklearn_linear` | `xgboost` | **C3R cascade** |
|---|---|---|---|
| Accuracy | 0.8183 | 0.8267 | **0.8267** |
| Macro-F1 | 0.8247 | 0.8338 | **0.8338** |
| Latency p50 | 1.0869 ms | 5.1284 ms | **2.1474 ms** |
| Latency p95 | 1.1450 ms | 5.4993 ms | **2.6216 ms** |
| Latency mean | 0.9466 ms | 4.9875 ms | **2.1978 ms** |
| Throughput | 1 056 /s | 200 /s | **455 /s** |
| Batch throughput | 55 405 /s | 11 653 /s | — |
| Memory (RSS) | — | — | **155.0 MB** |

**Derived results**
* **2.27× faster** than always-strong (−55.9% mean latency) at **identical** accuracy and macro-F1.
* **+0.0091 macro-F1** vs always-fast (0.8247 → 0.8338) for **2.32×** the latency — a deliberate, tunable trade-off.
* Routing distribution: **80.83% fast**, 1.67% escalated to strong, **17.50% abstain**; the strong backend is invoked on only **19.17%** of requests.
* Memory: 157.3 MB → 155.0 MB RSS across the benchmark run.

Machine-readable evidence: **[`benchmark.json`](benchmark.json)** (environment, versions, per-backend numbers, timestamp).

> ⚠️ **Read this before quoting the numbers.** The corpus is synthetic by design
> (so the system is reproducible offline). The numbers measure *the system*, not
> any real production dataset. The **relative** effects (latency saved at equal
> quality) are the transferable claim; absolute accuracy will differ on real data.

---

## 🧪 Verification

```bash
python -m pytest tests -q -W ignore::UserWarning
```

| Test file | Covers |
|---|---|
| `test_core.py` | error taxonomy, ENV overrides (incl. invalid input → `E100`) |
| `test_data.py` | deterministic generation, split accounting |
| `test_backends.py` | fit/predict, `Σ p = 1`, predict-before-fit → `E303` |
| `test_router.py` | routing policy incl. threshold boundaries, single-backend degradation, identity calibrator |
| `test_eval.py` | metric correctness, empty-input safety |
| `test_pipeline.py` | end-to-end train→evaluate→benchmark→JSON, and the innovation claim itself (cascade ≥ fast backend) |

---

## ⚠️ Limitations

* **Synthetic corpus** — no external dataset required, but absolute quality is not transferable.
* **Transformer path unmeasured** — implemented and import-guarded, but torch was not installed in this environment (⚪).
* **Abstention rate is high (17.5%)** on this corpus; `tau_low` must be tuned per domain.
* CPU-only, single-process; no request batching server.

## 🗺️ Roadmap

- [ ] Add a real-corpus loader (20 Newsgroups already wired) + publish that baseline too
- [ ] Exercise and benchmark the HuggingFace transformer tier on GPU
- [ ] Serve mode (FastAPI) exposing routing decisions as structured telemetry
- [ ] Cost-aware objective (price per 1k requests) alongside latency
- [ ] Persist calibrated routers so thresholds can be re-tuned without retraining backends

---

## 📁 Layout

```
intentforge/
  intentforge/
    core/         types · errors(E100–E500) · config(ENV) · interfaces(Protocol)
    data/         generator (seeded, offline) · loaders (builtin/csv/newsgroups)
    preprocess/   cleaning + TF-IDF factory
    domain/       backends (sklearn · xgboost · transformer) · router (C3R) · cascade
    hpo/          Optuna threshold tuning
    eval/         quality + latency + memory
    pipeline/     orchestration, reporting, benchmark.json
    cli.py        argparse entry (train/evaluate/benchmark/demo/predict)
    examples/     run_demo.py (end-to-end)
tests/            pytest suite
docs/             architecture.md
scripts/          bootstrap.py (one-click reproduction)
benchmark.json    committed benchmark evidence
requirements.txt  pinned key deps
requirements.lock.txt   full transitive lock
Dockerfile · Makefile · pytest.ini
```

## 📄 License

Apache-2.0. Built on scikit-learn (BSD-3), XGBoost (Apache-2.0), Optuna (MIT), psutil (BSD-3).

---

*晨星 · IntentForge*
