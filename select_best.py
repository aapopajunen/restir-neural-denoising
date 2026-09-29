"""Pick the best checkpoint(s) per run from the validation losses written by validate.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_JSONL_PATH = Path("metrics/validation_loss.jsonl")
DEFAULT_OUTPUT_JSON = Path("metrics/best_validation_models.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best checkpoint per run from validation-loss JSONL.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help="Validation-loss JSONL produced by validate.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="RecurrentL1Loss",
        help="Metric name to select.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="training_matched_validation",
        help="Protocol filter.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation",
        help="Dataset split filter.",
    )
    parser.add_argument(
        "--validation-nseqs",
        type=int,
        default=256,
        help="Validation sequence count filter.",
    )
    parser.add_argument(
        "--use-augment",
        action="store_true",
        help="Select augmented validation rows instead of deterministic ones.",
    )
    parser.add_argument(
        "--ema-label",
        action="append",
        dest="ema_labels",
        default=None,
        help="Optional EMA label filter. Can be given multiple times.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        default=None,
        help="Optional run-name filter. Can be given multiple times.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many best checkpoints to keep per run.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    filtered = []
    allowed_emas = None if args.ema_labels is None else set(args.ema_labels)
    allowed_runs = None if args.run_names is None else set(args.run_names)
    for row in rows:
        if row.get("metric") != args.metric:
            continue
        if row.get("protocol") != args.protocol:
            continue
        if row.get("dataset_split") != args.dataset_split:
            continue
        if bool(row.get("use_augment", False)) != bool(args.use_augment):
            continue
        if int(row.get("validation_nseqs_requested", -1)) != int(args.validation_nseqs):
            continue
        if allowed_emas is not None and row.get("ema_label") not in allowed_emas:
            continue
        if allowed_runs is not None and row.get("run_name") not in allowed_runs:
            continue
        filtered.append(row)
    return filtered


def select_best_per_run(rows: list[dict], *, top_k: int) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run_name"]].append(row)

    selected = {}
    for run_name, run_rows in grouped.items():
        run_rows.sort(
            key=lambda r: (
                float(r["mean"]),
                -int(r["kimg"]),
                str(r["snapshot"]),
            )
        )
        best_rows = []
        for row in run_rows[:top_k]:
            item = dict(row)
            best_rows.append(item)
        selected[run_name] = best_rows
    return selected


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.jsonl_path)
    rows = filter_rows(rows, args)
    if not rows:
        raise ValueError("No rows left after filtering")

    best_per_run = select_best_per_run(rows, top_k=args.top_k)
    summary = {
        "config": {
            "jsonl_path": str(args.jsonl_path),
            "metric": args.metric,
            "protocol": args.protocol,
            "dataset_split": args.dataset_split,
            "validation_nseqs": int(args.validation_nseqs),
            "use_augment": bool(args.use_augment),
            "ema_labels": None if args.ema_labels is None else list(args.ema_labels),
            "run_names": None if args.run_names is None else list(args.run_names),
            "top_k": int(args.top_k),
        },
        "best_per_run": best_per_run,
    }
    write_json(args.output_json, summary)

    print("\nBest checkpoint per run")
    for run_name in sorted(best_per_run):
        print(f"  {run_name}")
        for idx, row in enumerate(best_per_run[run_name], start=1):
            print(
                f"    {idx}. mean={float(row['mean']):.8f} "
                f"ema={row['ema_label']} kimg={int(row['kimg'])} snapshot={row['snapshot']}"
            )

    print(f"\nSaved selection summary to {args.output_json}")


if __name__ == "__main__":
    main()
