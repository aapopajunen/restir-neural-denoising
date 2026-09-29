"""Helpers shared by the evaluation scripts: locating datasets and reading run configs."""

import json
import os
import pickle
from pathlib import Path

import dnnlib
from dnnlib.util import EasyDict

#----------------------------------------------------------------------------

def get_data_root(data_root=None):
    data_root = data_root or os.environ.get('DENOISE_DATA_ROOT')
    if data_root is None:
        raise SystemExit('Dataset root not given: pass --data-root or set DENOISE_DATA_ROOT.')
    return Path(data_root)

#----------------------------------------------------------------------------
# Runs created by train.py store their config as config.json. Older runs
# launched through the Aalto cluster tooling store it in run_func_args.pkl,
# pickled with that tooling's EasyDict class; map it to dnnlib's.

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'EasyDict':
            return EasyDict
        return super().find_class(module, name)

def load_run_config(run_dir):
    """Return the training config of a run as an EasyDict with train.py's key names."""
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.json'
    if config_path.exists():
        with config_path.open() as f:
            return EasyDict(json.load(f))

    legacy_path = run_dir / 'run_func_args.pkl'
    if legacy_path.exists():
        with legacy_path.open('rb') as f:
            cfg = EasyDict(_LegacyUnpickler(f).load().config)
        cfg.loss = cfg.get('loss') or cfg.loss_kwargs['class_name']
        cfg.loss_scaling = cfg.get('ls', 1.0)
        cfg.train_data = cfg.get('data')
        return cfg

    raise FileNotFoundError(f'No config.json or run_func_args.pkl in {run_dir}')

def load_snapshot(path, device='cuda'):
    """Load the network from a snapshot pickle written by the training loop."""
    with dnnlib.util.open_url(str(path), verbose=False) as f:
        return pickle.load(f)['ema'].to(device).eval()

#----------------------------------------------------------------------------
