# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified from EDM2's train_edm2.py for recurrent denoising.

"""Train the recurrent denoiser.

Single GPU:
    python train.py --config configs/thesis_restir.json --data-root /path/to/datasets

Multiple GPUs:
    torchrun --standalone --nproc_per_node=4 train.py --config ... --data-root ...

Re-running with --resume <run dir> continues from the latest training-state-*.pt in it.
"""

import argparse
import json
import os
import re
import warnings
import torch
import dnnlib
from dnnlib.util import get_obj_by_name
from torch_utils import distributed as dist
import training.training_loop

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------

def get_data_root(data_root):
    data_root = data_root or os.environ.get('DENOISE_DATA_ROOT')
    if data_root is None:
        raise SystemExit('Dataset root not given: pass --data-root or set DENOISE_DATA_ROOT.')
    return data_root

def load_config(path):
    with open(path) as f:
        return dnnlib.EasyDict(json.load(f))

#----------------------------------------------------------------------------
# Translate a JSON config into arguments for training.training_loop.training_loop().

def setup_training_config(cfg, data_root, num_workers):
    c = dnnlib.EasyDict()

    c.train_ds_kwargs = [
        dnnlib.EasyDict(
            class_name  = 'training.dataset.MotionCompensatedDataset',
            path        = os.path.join(data_root, fname),
            fixed_crop  = cfg.get('fixed_crop', False),
            val_ratio   = 0.1,
            split       = 'all',
            crop_size   = cfg.train_crop_size,
            nframes     = cfg.train_seqlen,
        )
        for fname in cfg.train_data
    ]

    c.network_kwargs = dnnlib.EasyDict(
        class_name        = cfg.network,
        model_channels    = cfg.channels,
        dropout           = cfg.dropout,
        dtype             = cfg.dtype,
        input_buffers     = cfg.input_buffers,
        buffer_transforms = {k: get_obj_by_name(v) for k, v in cfg.buffer_transforms.items()},
    )
    c.loss_kwargs = dnnlib.EasyDict(class_name=cfg.loss)
    c.lr_kwargs   = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=cfg.lr, ref_batches=cfg.decay, rampup_Mimg=cfg.rampup)
    c.ema_kwargs  = dnnlib.EasyDict(class_name='training.phema.PowerFunctionEMA', stds=cfg.ema_stds)

    c.total_nimg      = cfg.duration
    c.batch_size      = cfg.batch
    c.batch_gpu       = cfg.get('batch_gpu') or None
    c.tbptt_step      = cfg.tbptt_step
    c.tbptt_window    = cfg.tbptt_window
    c.use_history     = cfg.get('use_history', True)
    c.use_gradscaler  = cfg.get('use_gradscaler', False)
    c.loss_scaling    = cfg.get('loss_scaling', 1)
    c.num_workers     = num_workers
    c.status_nimg     = cfg.get('status') or None
    c.snapshot_nimg   = cfg.get('snapshot') or None
    c.checkpoint_nimg = cfg.get('checkpoint') or None
    c.seed            = cfg.get('seed', 0)
    return c

#----------------------------------------------------------------------------

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if callable(obj):
        return f'{obj.__module__}.{obj.__name__}'
    return obj

def next_run_dir(outdir, desc):
    prev_run_ids = []
    if os.path.isdir(outdir):
        prev_run_ids = [re.match(r'^\d+', x) for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    cur_run_id = max([int(x.group()) for x in prev_run_ids if x is not None], default=-1) + 1
    return os.path.join(outdir, f'{cur_run_id:05d}-{desc}')

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',      help='JSON training config (see configs/). Not needed with --resume.')
    parser.add_argument('--data-root',   help='Directory containing the datasets named in the config. Default: $DENOISE_DATA_ROOT.')
    parser.add_argument('--outdir',      default='training-runs', help='Where to create the run directory.')
    parser.add_argument('--desc',        help='Run name suffix. Default: config file name.')
    parser.add_argument('--resume',      help='Existing run directory to continue.')
    parser.add_argument('--batch-gpu',   type=int, help='Override per-GPU batch size (uses gradient accumulation).')
    parser.add_argument('--num-workers', type=int, default=8, help='Data loader workers per accumulation round.')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if args.resume:
        run_dir = args.resume
        cfg = load_config(os.path.join(run_dir, 'config.json'))
    else:
        if not args.config:
            raise SystemExit('Either --config or --resume is required.')
        cfg = load_config(args.config)
        desc = args.desc or os.path.splitext(os.path.basename(args.config))[0]
        run_dir = next_run_dir(args.outdir, desc)
    if args.batch_gpu is not None:
        cfg.batch_gpu = args.batch_gpu

    c = setup_training_config(cfg, get_data_root(args.data_root), args.num_workers)

    if dist.get_rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'config.json'), 'wt') as f:
            json.dump(cfg, f, indent=4)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(make_json_serializable(c), f, indent=2)
        dist.print0(json.dumps(make_json_serializable(c), indent=2))
        dist.print0(f'Output directory: {run_dir}')
        dist.print0(f'Number of GPUs:   {dist.get_world_size()}')
    torch.distributed.barrier()

    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
