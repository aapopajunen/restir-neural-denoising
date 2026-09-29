"""Compute image metrics for sequences written by denoise.py.

Per frame: PSNR, SSIM and FLIP on Reinhard tone-mapped LDR images, and MAPE
on linear HDR radiance. One JSONL row is written per (metric, sequence) with
the per-frame values and their mean.

    python compute_metrics.py training-runs/00000-thesis_restir/test/network-snapshot-0033554-0.001

FLIP requires the flip-evaluator package (pip install flip-evaluator).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from sequence import Sequence
from torch_utils import misc

MAPE_EPS_SCALE = 0.01
METRICS = ["PSNR", "SSIM", "FLIP", "MAPE"]

#----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("denoised_dirs", nargs="+", type=Path, help="Output directories of denoise.py.")
    parser.add_argument("--jsonl-path", type=Path, default=Path("metrics/test_metrics.jsonl"), help="Output JSONL file.")
    parser.add_argument("--exposure", type=float, default=20.0, help="Exposure applied before Reinhard tone mapping.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute rows that already exist.")
    return parser.parse_args()

def row_key(row: dict) -> tuple:
    return (row["metric"], float(row["exposure"]), row["run_dir"], row["snapshot"], row["ds_name"], int(row["seq_idx"]))

def load_rows(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row_key(row): row for row in rows}

def write_rows(path: Path, rows: dict[tuple, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

#----------------------------------------------------------------------------

def tone_map_hwc(frame_hwc: torch.Tensor, exposure: float) -> torch.Tensor:
    chw = frame_hwc.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    return misc.reinhard(chw, exposure=exposure).clamp(0.0, 1.0)

def compute_sequence_metrics(seq: Sequence, *, exposure: float, psnr_metric, ssim_metric, flip) -> dict[str, list[float]]:
    values = {m: [] for m in METRICS}
    for frame in seq.frames:
        pred_hwc = frame["output"].to(torch.float32)
        target_hwc = frame["target"].to(torch.float32)

        pred_tm = tone_map_hwc(pred_hwc, exposure).to(psnr_metric.device)
        target_tm = tone_map_hwc(target_hwc, exposure).to(psnr_metric.device)
        values["PSNR"].append(float(psnr_metric(pred_tm, target_tm).item()))
        values["SSIM"].append(float(ssim_metric(pred_tm, target_tm).item()))

        pred_np = pred_tm[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        target_np = target_tm[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        _flip_map, flip_value, _params = flip.evaluate(target_np, pred_np, "LDR")
        values["FLIP"].append(float(flip_value))

        pred_chw = pred_hwc.permute(2, 0, 1).unsqueeze(0)
        target_chw = target_hwc.permute(2, 0, 1).unsqueeze(0)
        values["MAPE"].append(float(misc.mape(pred_chw, target_chw, eps_scale=MAPE_EPS_SCALE).mean().item()))
    return values

#----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    try:
        import flip_evaluator as flip
    except ImportError as exc:
        raise SystemExit("FLIP needs the flip-evaluator package: pip install flip-evaluator") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    rows = load_rows(args.jsonl_path)

    for denoised_dir in args.denoised_dirs:
        with (denoised_dir / "meta.json").open() as f:
            meta = json.load(f)

        for ds_dir in sorted(p for p in denoised_dir.iterdir() if p.is_dir()):
            ds_name = ds_dir.name
            seq_paths = sorted(ds_dir.glob(f"{ds_name}-*.pt"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
            for seq_path in seq_paths:
                seq_idx = int(seq_path.stem.rsplit("-", 1)[1])
                base = dict(exposure=args.exposure, run_dir=meta["run_dir"], snapshot=meta["snapshot"], ds_name=ds_name, seq_idx=seq_idx)
                if not args.overwrite and all(row_key(dict(base, metric=m)) in rows for m in METRICS):
                    continue

                print(f"{denoised_dir.name} | {ds_name} seq {seq_idx}")
                seq = Sequence.load(seq_path)
                metrics = compute_sequence_metrics(seq, exposure=args.exposure, psnr_metric=psnr_metric, ssim_metric=ssim_metric, flip=flip)
                for metric, per_frame in metrics.items():
                    row = dict(
                        base,
                        metric=metric,
                        metric_space="hdr" if metric == "MAPE" else "ldr",
                        path=str(seq_path),
                        nframes=len(per_frame),
                        mean=float(np.mean(per_frame)),
                        per_frame=per_frame,
                    )
                    rows[row_key(row)] = row

        write_rows(args.jsonl_path, rows)

    # Summary: mean over sequences per (snapshot, dataset, metric).
    summary = {}
    for row in rows.values():
        summary.setdefault((row["snapshot"], row["ds_name"], row["metric"]), []).append(row["mean"])
    print()
    for (snapshot, ds_name, metric), means in sorted(summary.items()):
        print(f"{snapshot:45s} {ds_name:32s} {metric:5s} {np.mean(means):.5f}")
    print(f"\nSaved metric rows to {args.jsonl_path}")

if __name__ == "__main__":
    main()
