"""IntentForge command-line entry point.

Usage
-----
    python -m intentforge.cli demo            # train + evaluate + benchmark
    python -m intentforge.cli train
    python -m intentforge.cli evaluate
    python -m intentforge.cli benchmark --out benchmark.json
    python -m intentforge.cli predict --text "check my balance" "cancel my plan"
"""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="intentforge", description="IntentForge: confidence-routed hybrid text classification."
    )
    p.add_argument("command", nargs="?", default="demo",
                   choices=["train", "evaluate", "benchmark", "demo", "predict"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--source", default="builtin", choices=["builtin", "csv", "newsgroups"])
    p.add_argument("--csv", default=None, help="path for --source csv")
    p.add_argument("--out", default="benchmark.json")
    p.add_argument("--transformer", action="store_true",
                   help="enable optional HuggingFace transformer backend")
    p.add_argument("--text", nargs="*", default=None)
    return p


def main(argv=None) -> int:  # pragma: no cover - CLI glue
    args = build_parser().parse_args(argv)

    from intentforge.core.config import Config
    from intentforge.pipeline.pipeline import IntentForgePipeline

    cfg = Config.from_env()
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.transformer:
        cfg.use_transformer = True

    loader_kwargs = {}
    if args.source == "csv":
        if not args.csv:
            print("ERROR: --source csv requires --csv <path>", file=sys.stderr)
            return 2
        loader_kwargs = {"path": args.csv}

    pipe = IntentForgePipeline(cfg, data_source=args.source, **loader_kwargs)
    cmd = args.command

    if cmd == "train":
        pipe.train()
        print("[intentforge] training complete")
        return 0
    if cmd == "evaluate":
        pipe.train()
        pipe.evaluate()
        pipe.print_report()
        return 0
    if cmd == "benchmark":
        pipe.train()
        pipe.evaluate()
        pipe.benchmark()
        pipe.print_report()
        pipe.save_benchmark(args.out)
        return 0
    if cmd == "predict":
        pipe.train()
        texts = args.text or ["hello there", "cancel my subscription now"]
        for r in pipe.predict(texts):
            print(json.dumps(r, ensure_ascii=False))
        return 0

    # default: demo
    pipe.train()
    pipe.evaluate()
    pipe.benchmark()
    pipe.print_report()
    pipe.save_benchmark(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
