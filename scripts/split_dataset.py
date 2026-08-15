"""Split an enriched dataset into dev/test, stratified by category_macro.

Usage:
    python scripts/split_dataset.py --data data/dataset_100.enriched.jsonl
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.enrich import load_enriched_jsonl, save_enriched_jsonl
from vdriftbench.splits import split_summary, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified dev/test split by category_macro")
    parser.add_argument("--data", default="data/dataset_100.enriched.jsonl")
    parser.add_argument("--dev-out", default="data/dev.jsonl")
    parser.add_argument("--test-out", default="data/test.jsonl")
    parser.add_argument("--dev-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = load_enriched_jsonl(args.data)
    dev, test = stratified_split(samples, dev_ratio=args.dev_ratio, seed=args.seed)

    save_enriched_jsonl(dev, args.dev_out)
    save_enriched_jsonl(test, args.test_out)

    print(json.dumps(split_summary(dev, test), ensure_ascii=False, indent=2))
    print(f"dev -> {args.dev_out}")
    print(f"test -> {args.test_out}")


if __name__ == "__main__":
    main()
