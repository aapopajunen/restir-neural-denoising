"""Compute validation losses for the snapshots of one or more training runs.

Uses the same protocol as training: the run's validation datasets cropped to
128x128, sequences processed with the run's TBPTT windowing, and the run's
training loss. Results are appended to a JSONL file (one row per snapshot);
snapshots already present in the file are skipped, so the script can be
re-run as training progresses. Use select_best.py to pick checkpoints.

    python validate.py training-runs/00000-thesis_restir --data-root /path/to/datasets
"""

from __future__ import annotations

import argparse
import json
from itertools import chain
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

import dnnlib
from dnnlib.util import EasyDict
from run_utils import get_data_root, load_run_config, load_snapshot
from torch_utils import misc
from training.augmentation import SequenceAugment
from training.training_loop import _on_new_sequence, _preprocess_sample, make_augs

PROTOCOL = "training_matched_validation"
CROP_SIZE = 128

#----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Training run directories.")
    parser.add_argument("--data-root", help="Directory containing the datasets. Default: $DENOISE_DATA_ROOT.")
    parser.add_argument("--jsonl-path", type=Path, default=Path("metrics/validation_loss.jsonl"), help="Output JSONL file.")
    parser.add_argument("--validation-nseqs", type=int, default=256, help="Cropped validation sequences per snapshot. Must be a multiple of --batch-gpu.")
    parser.add_argument("--ema-label", action="append", dest="ema_labels", default=None, help="Only evaluate these EMA labels, e.g. ema-0.001. Repeatable.")
    parser.add_argument("--batch-gpu", type=int, default=16, help="Sequences per batch.")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--use-augment", action="store_true", help="Apply the training augmentations during validation.")
    parser.add_argument("--top-k", type=int, default=5, help="Best rows to print per run at the end.")
    return parser.parse_args()

#----------------------------------------------------------------------------

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def row_key(row: dict) -> tuple:
    return (
        row["metric"],
        row["protocol"],
        row["run_name"],
        row["ema_label"],
        int(row["kimg"]),
        row["snapshot"],
        int(row["validation_nseqs_requested"]),
        bool(row.get("use_augment", False)),
    )

def iter_jobs(run_dirs: list[Path], ema_labels: list[str] | None):
    """Yield (run_dir, ema_label, snapshot record), interleaving runs and bisecting
    each run's snapshot list so partial results cover the whole training range."""
    per_run = {p: misc.select_variants_from_pickles(p) for p in run_dirs}
    all_emas = sorted(set(chain.from_iterable(d.keys() for d in per_run.values())))
    if ema_labels:
        all_emas = [e for e in all_emas if e in set(ema_labels)]

    balanced = {(p, ema): list(misc.balanced_iter(per_run[p].get(ema, []))) for p in run_dirs for ema in all_emas}
    max_len = max((len(b) for b in balanced.values()), default=0)
    for idx in range(max_len):
        for ema in all_emas:
            for p in run_dirs:
                b = balanced[(p, ema)]
                if idx < len(b):
                    yield p, ema, b[idx]

#----------------------------------------------------------------------------

def build_run_context(run_dir: Path, data_root: Path, args) -> dict:
    cfg = load_run_config(run_dir)

    val_datasets = [
        dnnlib.util.construct_class_by_name(
            class_name="training.dataset.MotionCompensatedDataset",
            path=str(data_root / fname),
            fixed_crop=bool(cfg.get("fixed_crop", False)),
            val_ratio=0.0,
            crop_size=CROP_SIZE,
            split="all",
            nframes=int(cfg.train_seqlen),
        )
        for fname in cfg.val_data
    ]
    nframes = int(val_datasets[0].get_nframes())
    if any(int(ds.get_nframes()) != nframes for ds in val_datasets):
        raise ValueError(f"Validation datasets for {run_dir} have inconsistent nframes")

    val_ds = ConcatDataset(val_datasets)
    loss_fn = dnnlib.util.construct_class_by_name(class_name=cfg.loss).to("cuda").eval()

    return dict(
        val_ds=val_ds,
        val_datasets=[str(data_root / fname) for fname in cfg.val_data],
        nframes=nframes,
        nseqs=len(val_ds) // nframes,
        loss_fn=loss_fn,
        metric=loss_fn.__class__.__name__,
        tbptt_step=int(cfg.tbptt_step),
        tbptt_window=int(cfg.tbptt_window),
        use_history=bool(cfg.get("use_history", True)),
        loss_scaling=float(cfg.get("loss_scaling", 1.0)),
        data_loader_kwargs=dict(class_name="torch.utils.data.DataLoader", num_workers=args.num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True),
    )

def evaluate_snapshot(snapshot_path: Path, ctx: dict, *, validation_nseqs: int, batch_gpu: int, use_augment: bool) -> dict:
    if validation_nseqs % batch_gpu != 0:
        raise ValueError(f"--validation-nseqs={validation_nseqs} must be a multiple of --batch-gpu={batch_gpu}")
    if ctx["nframes"] % ctx["tbptt_step"] != 0:
        raise ValueError(f"nframes={ctx['nframes']} must be divisible by tbptt_step={ctx['tbptt_step']}")
    total_windows = (validation_nseqs // batch_gpu) * (ctx["nframes"] // ctx["tbptt_step"])

    dataset_sampler = misc.RoundRobinSampler(
        nseqs=ctx["nseqs"],
        nframes=ctx["nframes"],
        batch_size=batch_gpu,
        rank=0,
        world_size=1,
        shuffle=True,
        split=1,
        split_idx=0,
        seed=0,
    )
    val_loader = dnnlib.util.construct_class_by_name(dataset=ctx["val_ds"], batch_sampler=dataset_sampler, **ctx["data_loader_kwargs"])
    net = load_snapshot(snapshot_path)
    losses = []

    try:
        with torch.inference_mode():
            tbptt = misc.TBPTT(
                step_size=ctx["tbptt_step"],
                window_len=ctx["tbptt_window"],
                device="cuda",
                iterator=iter(val_loader),
                preprocess=_preprocess_sample,
                on_new_sequence=_on_new_sequence,
                state=EasyDict(augment=SequenceAugment(seed=0, augs=make_augs() if use_augment else []), h=None, device="cuda"),
            )
            for window_idx in range(total_windows):
                print(f"  window {window_idx + 1}/{total_windows}", end="\r")
                preds, refs = [], []
                for x in tbptt.step():
                    y, tbptt.state.h = net(x, tbptt.state.h if ctx["use_history"] else None)
                    preds.append(y.to(torch.float32))
                    refs.append(x)
                losses.append(float(ctx["loss_fn"](preds, refs).float().item()))
    finally:
        del val_loader
        del net
    print()

    mean = float(sum(losses) / len(losses))
    return dict(mean=mean, scaled_mean=mean * ctx["loss_scaling"], total_window_losses=total_windows, per_window=losses)

#----------------------------------------------------------------------------

def print_best(rows: list[dict], top_k: int) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["run_name"], []).append(row)
    for run_name in sorted(grouped):
        grouped[run_name].sort(key=lambda r: (float(r["mean"]), int(r["kimg"]), str(r["snapshot"])))
        print(f"\n{run_name}")
        for idx, row in enumerate(grouped[run_name][:top_k], start=1):
            print(f"  {idx:>2}. {row['metric']} mean={float(row['mean']):.8f} kimg={int(row['kimg'])} snapshot={row['snapshot']}")

def main() -> None:
    args = parse_args()
    data_root = get_data_root(args.data_root)
    existing_keys = {row_key(row) for row in load_rows(args.jsonl_path)}
    run_contexts: dict[Path, dict] = {}

    for run_dir, ema_label, variant in iter_jobs(args.run_dirs, args.ema_labels):
        if run_dir not in run_contexts:
            print(f"Loading run context for {run_dir.name}")
            run_contexts[run_dir] = build_run_context(run_dir, data_root, args)
        ctx = run_contexts[run_dir]

        row = {
            "metric": ctx["metric"],
            "protocol": PROTOCOL,
            "dataset_split": "validation",
            "run_name": run_dir.name,
            "run_path": str(run_dir),
            "ema_label": str(ema_label),
            "kimg": int(variant.kimg),
            "snapshot": str(variant.snapshot),
            "validation_nseqs_requested": int(args.validation_nseqs),
            "use_augment": bool(args.use_augment),
        }
        if row_key(row) in existing_keys:
            print(f"Skipping existing row for {run_dir.name} {variant.snapshot}")
            continue

        print(f"[{run_dir.name}] {variant.snapshot}")
        result = evaluate_snapshot(run_dir / variant.snapshot, ctx, validation_nseqs=args.validation_nseqs, batch_gpu=args.batch_gpu, use_augment=args.use_augment)
        row.update(
            mean=result["mean"],
            scaled_mean=result["scaled_mean"],
            total_window_losses=result["total_window_losses"],
            batch_gpu=args.batch_gpu,
            crop_size=CROP_SIZE,
            tbptt_step=ctx["tbptt_step"],
            tbptt_window=ctx["tbptt_window"],
            use_history=ctx["use_history"],
            val_datasets=ctx["val_datasets"],
            per_window=result["per_window"],
        )
        append_jsonl(args.jsonl_path, row)
        existing_keys.add(row_key(row))
        print(f"  {ctx['metric']} mean={row['mean']:.8f}")

    print_best(load_rows(args.jsonl_path), top_k=args.top_k)

if __name__ == "__main__":
    main()
