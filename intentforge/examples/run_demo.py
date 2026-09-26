"""End-to-end demo.

Trains on the synthetic intent dataset, evaluates every backend plus the C3R
cascade, runs a real latency benchmark on CPU, prints a report and writes
``benchmark.json`` (evidence of a genuine end-to-end run).

Run:
    python intentforge/examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

# make the package importable when executed directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intentforge.core.config import Config
from intentforge.pipeline.pipeline import IntentForgePipeline


def main(out_path: str = "benchmark.json"):
    cfg = Config()
    pipe = IntentForgePipeline(cfg, data_source="builtin")
    pipe.train()
    pipe.evaluate()
    pipe.benchmark()
    pipe.print_report()
    pipe.save_benchmark(out_path)

    # quick live-predict sanity check
    samples = [
        "what is my current balance",
        "i want to cancel my subscription and talk to a human about a wrong charge",
    ]
    print("\n=== Live predict (sample) ===")
    for r in pipe.predict(samples):
        print(f"  [{r['backend']:<14}] {r['label']:<20} cal_conf={r['calibrated_confidence']:.3f} "
              f"abstain={r['abstain']} ({r['reason']})")
    return pipe.benchmark


if __name__ == "__main__":
    main()
