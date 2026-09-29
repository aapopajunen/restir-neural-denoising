"""Denoise full-resolution test sequences with a trained snapshot.

For every sequence in each dataset, writes
    <outdir>/<dataset>/<dataset>-<seq_idx>.pt
a Sequence (see sequence.py) whose frames hold 'output' (denoised radiance),
'target' (reference radiance) and 'blend_weight' (the recurrent blend map),
all in the dataset's normalized radiance units. compute_metrics.py consumes
this directory.

    python denoise.py training-runs/00000-thesis_restir \\
        --snapshot network-snapshot-0033554-0.001.pkl --data-root /path/to/datasets

Datasets default to the run config's test_data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dnnlib.util import EasyDict
from run_utils import get_data_root, load_run_config, load_snapshot
from sequence import Sequence
from torch_utils import misc
from training.dataset import MotionCompensatedDataset
from training.training_loop import to_fp32

THESIS_TEST_DATA = [
    "165-bistro-exterior-test/dataset.hdf5",
    "166-bistro-interior-test/dataset.hdf5",
    "167-emerald-square-day-test/dataset.hdf5",
    "175-zero-day-test/dataset.hdf5",
    "169-veach-ajar-test/dataset.hdf5",
]

#----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Training run directory.")
    parser.add_argument("--snapshot", required=True, help="Snapshot path, relative to the run directory.")
    parser.add_argument("--data-root", help="Directory containing the datasets. Default: $DENOISE_DATA_ROOT.")
    parser.add_argument("--dataset", action="append", dest="datasets", default=None, help="Dataset (relative to the data root) to denoise. Repeatable. Default: the config's test_data.")
    parser.add_argument("--outdir", type=Path, help="Output directory. Default: <run_dir>/test/<snapshot name>.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render datasets whose outputs already exist.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data_root = get_data_root(args.data_root)
    cfg = load_run_config(args.run_dir)
    datasets = args.datasets or cfg.get("test_data") or THESIS_TEST_DATA
    use_history = bool(cfg.get("use_history", True))

    snapshot_path = args.run_dir / args.snapshot
    outdir = args.outdir or args.run_dir / "test" / Path(args.snapshot).stem
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "meta.json").open("w") as f:
        json.dump(dict(run_dir=str(args.run_dir), snapshot=args.snapshot, datasets=datasets), f, indent=2)

    net = load_snapshot(snapshot_path)
    print(f"Loaded {snapshot_path}")

    with torch.inference_mode():
        for ds_fname in datasets:
            ds_name = Path(ds_fname).parent.name
            ds = MotionCompensatedDataset(path=str(data_root / ds_fname), split="all", crop_size=None, val_ratio=0.1)
            loader = DataLoader(dataset=ds, num_workers=0, batch_size=1)
            expected = [outdir / ds_name / f"{ds_name}-{i}.pt" for i in range(ds.nsamples)]
            if not args.overwrite and all(p.exists() for p in expected):
                print(f"Skipping {ds_name}: outputs exist")
                continue

            print(f"Denoising {ds_name} ({ds.nsamples} sequences x {ds.nframes} frames)")
            seq = Sequence()
            h = None
            for i, batch in enumerate(loader):
                batch = to_fp32(misc.move_to_device(batch, "cuda"))
                if int(batch["frame_idx"][0]) == 0:
                    h = None
                y, h = net(batch, h if use_history else None)
                seq.append(EasyDict(
                    output=y,
                    target=batch["buffers"]["target"],
                    blend_weight=h["blend_weights"],
                ))
                if (i + 1) % ds.nframes == 0:
                    seq.save(expected[i // ds.nframes])
                    seq = Sequence()

    print(f"Saved to {outdir}")

if __name__ == "__main__":
    main()
