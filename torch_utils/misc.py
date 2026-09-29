# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import contextlib
import math
import os
import pickle
import re
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import torch
import torch.nn.functional as F

import dnnlib
from dnnlib.util import EasyDict, get_obj_by_name

Reduction = Literal["none", "mean", "sum"]

#----------------------------------------------------------------------------
# Re-seed torch & numpy random generators based on the given arguments.

def set_random_seed(*args):
    seed = hash(args) % (1 << 31)
    torch.manual_seed(seed)
    np.random.seed(seed)

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator




MASK64 = 0xFFFFFFFFFFFFFFFF
PHI64  = 0x9e3779b97f4a7c15


def _mix64(x):
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & MASK64
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb & MASK64
    x = x ^ (x >> 31)
    return x & MASK64


def _f_fn(right, round_no, key, mod):
    return _mix64((right + key + PHI64 * round_no) & MASK64) % mod


def _factor_pair(n):
    if n <= 1:
        return (1, 1)
    a = int(math.isqrt(n))
    while a > 1 and n % a != 0:
        a -= 1
    if n % a != 0:
        a = 1
    b = n // a
    return (a, b)


def feistel_permute_on_N(k, N, key=0, rounds=8):
    if N <= 1:
        return 0
    A, B = _factor_pair(N)
    L = k % A
    R = k // A
    aA, aB = A, B
    for r in range(rounds):
        L, R = R, (L + _f_fn(R, r, key, aA)) % aA
        aA, aB = aB, aA
    return L + (aA if rounds % 2 == 0 else aB) * R


def _coprime_step(n, s):
    if n <= 2: return 1
    a = (_mix64(s) % (n - 1)) + 1
    while math.gcd(a, n) != 1:
        a = (a + 1) % n or 1
    return a


def permute(idx, N, start_seed=12345, rounds=8):
    if N == 0: return 0
    epoch = idx // N
    i = idx % N                         # << important
    seed = (start_seed + epoch) & MASK64

    key = _mix64(seed)
    a = _coprime_step(N, seed ^ 0xdeadbeef)
    b = _mix64(seed ^ 0xabcdef01) % N

    x = feistel_permute_on_N(i, N, key=key, rounds=rounds)
    return (b + a * x) % N


class RoundRobinSampler(torch.utils.data.Sampler):
    def __init__(self,
                 nseqs: int,
                 nframes: int,
                 batch_size: int,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 seed: int = 0,
                 split: int = 1,
                 split_idx: int = 0,
                 start_idx: int = 0):
        self.N          = nseqs
        self.T          = nframes
        self.B          = batch_size // world_size
        self.rank       = rank
        self.world_size = world_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.start_idx  = start_idx + rank * self.T
        self.split      = split
        self.split_idx  = split_idx

    def permute(self, idx):
        seq_idx      = (idx // self.T) % self.N
        frame_idx    = idx % self.T
        return permute(seq_idx, self.N, self.seed) * self.T + frame_idx

    def get_batch_indices(self, idx):
        start  = idx
        end    = idx + self.world_size * self.T * self.B
        stride = self.world_size * self.T
        return map(self.permute, range(start, end, stride))

    # ──────────────────────────────────────────────────────────────────────
    #  Infinite iterator
    # ──────────────────────────────────────────────────────────────────────
    def __iter__(self):
        idx        = self.start_idx
        self.epoch = None

        while True:
            # Compute batch indices
            slice_size = self.B // self.split
            s = self.split_idx * slice_size
            e = (self.split_idx + 1) * slice_size
            yield list(self.get_batch_indices(idx))[s:e]

            # Update index
            idx += 1

            if idx % self.T == 0:
                idx = ((idx - self.T) + self.world_size * self.T * self.B)# % (self.N * self.T) # Jump to next set of sequences


class TBPTT():
    def __init__(self, step_size, window_len, device, iterator, preprocess, on_new_sequence, state):
        self.step_size = step_size
        self.window_len = window_len
        self.device = device
        self.iterator = iterator
        self.preprocess = preprocess
        self.on_new_sequence = on_new_sequence
        self.state = state
        self.rb = deque(maxlen=window_len)

    def step(self):
        # Fill ring buffer
        for _ in range(self.step_size):
            x = next(self.iterator)
            x = move_to_device(x, self.device)

            if x['frame_idx'][0] == 0:
                self.rb.clear()
                self.on_new_sequence(self.state)

            self.rb.append(self.preprocess(x, self.state))

        # This happens only during start of new sequences
        while len(self.rb) != self.rb.maxlen:
            x = next(self.iterator)
            x = move_to_device(x, self.device)
            x = self.preprocess(x, self.state)
            self.rb.append(x)

        return self.rb


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings('ignore', '`data_source` argument is not used and will be removed')
        super().__init__(dataset)
        self.dataset_size = len(dataset)
        self.start_idx    = start_idx + rank
        self.stride       = num_replicas
        self.shuffle      = shuffle
        self.seed         = seed

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                # order = np.arange(self.dataset_size)
                # if self.shuffle:
                #     np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(order)
            # yield int(order[idx % self.dataset_size])
            yield int(permute(idx % self.dataset_size, self.dataset_size, hash((self.seed, epoch)) % (1 << 31)))
            idx += self.stride


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name])


@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname


@torch.no_grad()
def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()


def move_to_device(obj, device):
    """
    Recursively moves all torch.Tensor objects in a nested structure to the given device.
    
    Args:
        obj: The input data structure (dict, list, tuple, tensor, etc.)
        device: The torch device (e.g., "cuda", "cpu", torch.device object)

    Returns:
        A new data structure with tensors moved to the specified device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Preserve type (tuple or list)
        cls = type(obj)
        return cls(move_to_device(v, device) for v in obj)
    else:
        return obj  # Leave other types unchanged
    
import torch


def to_dtype(obj, src_dtype, dst_dtype):
    """
    Recursively casts all torch.Tensor objects with dtype == src_dtype
    to dst_dtype within a nested structure.

    Args:
        obj: Nested structure (dict, list, tuple, tensor, etc.)
        src_dtype: Source dtype to look for (e.g., torch.float16)
        dst_dtype: Target dtype to cast to (e.g., torch.float32)

    Returns:
        New structure with matching tensors cast to dst_dtype.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(dst_dtype) if obj.dtype == src_dtype else obj
    elif isinstance(obj, dict):
        return {k: to_dtype(v, src_dtype, dst_dtype) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(to_dtype(v, src_dtype, dst_dtype) for v in obj)
    else:
        return obj


def detach_tensors(obj):
    """
    Recursively detaches all torch.Tensor objects in a nested structure 
    from the computation graph.

    Args:
        obj: The input data structure (dict, list, tuple, tensor, etc.)

    Returns:
        A new data structure with tensors detached.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(detach_tensors(v) for v in obj)
    else:
        return obj  # Leave other types unchanged


def clone(obj):
    """
    Recursively clones all torch.Tensor objects in a nested structure.

    Args:
        obj: The input data structure (dict, list, tuple, tensor, etc.)

    Returns:
        A new data structure with cloned tensors and the same structure.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: clone(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(clone(v) for v in obj)
    else:
        return obj  # Leave non-tensors unchanged


def get_grid(flow):
    _,_,H,W = flow.shape

    # Build base pixel-center grid in float32 for precision
    grid_dtype = torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=grid_dtype),
        torch.arange(W, device=flow.device, dtype=grid_dtype),
        indexing="ij"
    )
    xx = xx + 0.5
    yy = yy + 0.5

    # Cast flow to grid dtype for math; keep src dtype for sampling
    flow32 = flow.to(grid_dtype)

    # Normalise to [-1,1] for align_corners=False
    x_sample = (xx + flow32[:, 0]) / W * 2 - 1
    y_sample = (yy + flow32[:, 1]) / H * 2 - 1
    
    return  torch.stack((x_sample, y_sample), dim=-1)  # (N,H,W,2)


def reproj(src, flow, interpolation="bilinear", padding_mode="zeros", return_ob_mask=False):
    """
    Warp `src` with a backward flow field (Δx, Δy) in pixel units.

    src  : (..., C, H, W) float tensor
    flow : (..., 2, H, W) float tensor, backward flow in pixels
           flow[..., 0, y, x] = Δx, flow[..., 1, y, x] = Δy

    returns: (..., C, H, W)
    """
    if src.ndim < 4 or flow.ndim < 4 or flow.shape[-3] != 2:
        raise ValueError("src (...,C,H,W) and flow (...,2,H,W) required")
    if src.shape[:-3] != flow.shape[:-3] or src.shape[-2:] != flow.shape[-2:]:
        raise ValueError("Leading dims and HxW of src and flow must match")

    if not src.is_floating_point() or not flow.is_floating_point():
        raise TypeError("src and flow must be floating tensors")

    lead = src.shape[:-3]
    N = math.prod(lead) if lead else 1
    C, H, W = src.shape[-3:]

    src_flat  = src.reshape(N, C, H, W)
    flow_flat = flow.reshape(N, 2, H, W)

    # Build base pixel-center grid in float32 for precision
    grid_dtype = torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(H, device=src.device, dtype=grid_dtype),
        torch.arange(W, device=src.device, dtype=grid_dtype),
        indexing="ij"
    )
    xx = xx + 0.5
    yy = yy + 0.5

    # Cast flow to grid dtype for math; keep src dtype for sampling
    flow32 = flow_flat.to(grid_dtype)

    # Normalise to [-1,1] for align_corners=False
    x_sample = (xx + flow32[:, 0]) / W * 2 - 1
    y_sample = (yy + flow32[:, 1]) / H * 2 - 1
    grid = torch.stack((x_sample, y_sample), dim=-1)  # (N,H,W,2)

    warped = F.grid_sample(
        src_flat, grid.to(src_flat.dtype) if src_flat.dtype != torch.float32 else grid,
        mode=interpolation, padding_mode=padding_mode, align_corners=False
    )
    
    if return_ob_mask:
        ob_mask = (grid[..., 0].abs() > 1) | (grid[..., 1].abs() > 1)
        ob_mask = ob_mask.unsqueeze(1)  # (N,1,H,W)
        return warped.reshape(*lead, C, H, W), ob_mask.reshape(*lead, 1, H, W)
    else:
        return warped.reshape(*lead, C, H, W)


def reinhard(x, exposure=1.0):
    scaled = (x * exposure).clamp(0, torch.inf)
    return (scaled / (scaled + 1)) ** 2.2


def transform(x, buffer_transforms):
    for buffer in x['buffers'].keys():
        if buffer in buffer_transforms:
            transform = buffer_transforms[buffer]
            if isinstance(transform, str):
                transform = get_obj_by_name(transform)
            x['buffers'][buffer] = transform(x['buffers'][buffer])
    return x


def get_disocclusion_mask(
    wpos,            # [B,3,H,W] current world-space positions
    prev_wpos=None,  # [B,3,H,W] previous world-space positions (or None for first frame)
    motion_vector=None,    # [B,2,H,W] motion vectors in current frame
    crop_offset=None,      # [...,2] current crop offset (broadcastable to B)
    prev_crop_offset=None, # [...,2] previous crop offset (same shape as crop_offset)
    cam=None,
    rel_eps: float = 1e-6,
):
    """
    Computes a normalized relative depth ratio between current and previous frame:
        R = depth_prev / depth_curr
      where:
        R ≈ 1 → same surface
        R < 1 → disocclusion (current surface is farther)
        R > 1 → occlusion   (current surface is closer)

    - Out-of-bounds (ob_mask) → 0
    - All outputs are finite (no NaNs/Infs)
    - If prev_wpos is None → everything disoccluded (returns zeros)

    Args:
        wpos:             [B,3,H,W] world-space positions for current frame.
        prev_wpos:        [B,3,H,W] world-space positions for previous frame, or None.
        motion_vector:    [B,2,H,W] backward flow (current → previous) in pixels.
        crop_offset:      [...,2] crop offset for current frame.
        prev_crop_offset: [...,2] crop offset for previous frame.
        cam_position:     [B,3] camera position.
        cam_target:       [B,3] camera target point.
        rel_eps:          small epsilon to prevent divide-by-zero.

    Returns:
        ratio_clean : [B,1,H,W] float32 tensor, finite in [0, +∞).
    """
    B, _, H, W = wpos.shape
    device, dtype = wpos.device, wpos.dtype

    # Handle first frame (no previous)
    if prev_wpos is None:
        return torch.zeros((B, 1, H, W), dtype=dtype, device=device)

    # Flow + crop offset alignment
    crop_diff = (prev_crop_offset - crop_offset).flip(-1)  # swap xy -> yx (view-like)
    crop_diff = crop_diff.unsqueeze(-1).unsqueeze(-1)      # (..., 2, 1, 1), broadcast over H,W
    flow = motion_vector - crop_diff

    # Warp previous world positions into current pixel grid (using backward flow)
    warped_wpos, ob_mask = reproj(prev_wpos, flow, interpolation='nearest', return_ob_mask=True)
    if ob_mask.ndim == 3:
        ob_mask = ob_mask.unsqueeze(1)  # [B,1,H,W]

    # Camera basis
    cam_pos = cam['position'].view(B, 3, 1, 1)
    cam_fwd = (cam['target'] - cam['position']).view(B, 3, 1, 1)  # should be unit vector into scene

    # Paranoia normalization: avoid division by zero or NaN propagation
    norm = cam_fwd.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cam_fwd = cam_fwd / norm

    # Linear view-space depths (dot with camera forward)
    depth_curr = ((wpos        - cam_pos) * cam_fwd).sum(dim=1, keepdim=True)   # [B,1,H,W]
    depth_prev = ((warped_wpos - cam_pos) * cam_fwd).sum(dim=1, keepdim=True)   # [B,1,H,W]

    # Safe division for ratio
    denom = torch.clamp(depth_curr, min=rel_eps)
    ratio = depth_prev / denom

    # Valid if finite and positive
    valid = torch.isfinite(ratio) & (depth_curr > 0) & (depth_prev > 0)

    env_hit_prev = (warped_wpos == 0).all(dim=1, keepdim=True)
    env_hit      = (wpos        == 0).all(dim=1, keepdim=True)

    # Clean: 1 for invalid, 0 for OB / new env hits
    ratio_clean = torch.where(valid, ratio, torch.ones_like(ratio))
    ratio_clean = torch.where(ob_mask.bool(), torch.zeros_like(ratio_clean), ratio_clean)
    ratio_clean = torch.where((env_hit & ~env_hit_prev), torch.zeros_like(ratio_clean), ratio_clean)

    return ratio_clean  # [B,1,H,W]


def masked_reproject_gather(
    x,                     # [B,C,H,W] feature/color to reproject
    wpos,                  # [B,3,H,W] current-frame world positions
    prev_wpos=None,        # [B,3,H,W] prev-frame world positions
    motion_vector=None,    # [B,2,H,W] backward motion vectors (Δx, Δy in pixels)
    crop_offset=None,      # [...,2]
    prev_crop_offset=None, # [...,2]
    cam=None,
    rel_threshold: float = 0.01,
    rel_eps: float = 1e-6,
):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # First frame → no reprojection available
    if prev_wpos is None:
        out = torch.zeros_like(x)
        disocc4 = torch.ones((B, 4, H, W), dtype=torch.bool, device=device)
        return out, disocc4

    # --------------------------------------------------------------------------
    # 1. ALIGN FLOW (crop-aware)
    # --------------------------------------------------------------------------
    if crop_offset is not None and prev_crop_offset is not None:
        crop_diff = (prev_crop_offset - crop_offset).flip(-1)
        crop_diff = crop_diff.unsqueeze(-1).unsqueeze(-1)   # [...,2,1,1]
        flow = motion_vector - crop_diff                    # [B,2,H,W]
    else:
        flow = motion_vector                                # [B,2,H,W]

    flow_x = flow[:, 0]  # [B,H,W]
    flow_y = flow[:, 1]  # [B,H,W]

    # --------------------------------------------------------------------------
    # 2. CONTINUOUS PREV-FRAME COORDS + 4 NEIGHBOR PIXELS
    #    We treat pixel centers at (j + 0.5, i + 0.5) and add flow in pixels.
    # --------------------------------------------------------------------------
    grid_dtype = torch.float32  # for stability
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=grid_dtype),
        torch.arange(W, device=device, dtype=grid_dtype),
        indexing="ij"
    )
    xx = xx + 0.5
    yy = yy + 0.5

    # continuous prev-frame positions (pixel-center coords)
    x_prev = xx.unsqueeze(0) + flow_x.to(grid_dtype)   # [B,H,W]
    y_prev = yy.unsqueeze(0) + flow_y.to(grid_dtype)   # [B,H,W]

    # convert to "pixel index space": centers at k+0.5, so index = coord - 0.5
    x_idx = x_prev - 0.5
    y_idx = y_prev - 0.5

    x0 = torch.floor(x_idx)
    y0 = torch.floor(y_idx)
    x1 = x0 + 1
    y1 = y0 + 1

    # fractional part in that index space
    fx = (x_idx - x0).to(dtype)   # [B,H,W]
    fy = (y_idx - y0).to(dtype)   # [B,H,W]

    # Stack 4 neighbors: TL, TR, BL, BR
    # 0: (x0,y0), 1: (x1,y0), 2: (x0,y1), 3: (x1,y1)
    x_stack = torch.stack([x0, x1, x0, x1], dim=1)  # [B,4,H,W]
    y_stack = torch.stack([y0, y0, y1, y1], dim=1)  # [B,4,H,W]

    # out-of-bounds BEFORE clamping
    oob = (
        (x_stack < 0) | (x_stack > W - 1) |
        (y_stack < 0) | (y_stack > H - 1)
    )  # [B,4,H,W]

    # clamp to valid index range for safe gather
    x_clamp = x_stack.clamp(0, W - 1).long()
    y_clamp = y_stack.clamp(0, H - 1).long()

    # --------------------------------------------------------------------------
    # 3. GATHER 4 NEIGHBORS FROM [prev_wpos, x]
    # --------------------------------------------------------------------------
    stacked = torch.cat([prev_wpos, x], dim=1)      # [B,3+C,H,W]
    BC = stacked.shape[1]
    N = H * W

    stacked_flat = stacked.view(B, BC, N)           # [B,3+C,N]

    # linear indices over spatial dimension
    idx = (y_clamp * W + x_clamp).view(B, 4, N)     # [B,4,N]

    # expand for gather: [B,4,3+C,N]
    stacked_b = stacked_flat.unsqueeze(1).expand(-1, 4, -1, -1)
    idx_b     = idx.unsqueeze(2).expand(-1, -1, BC, -1)

    gathered_flat = torch.gather(stacked_b, 3, idx_b)  # [B,4,3+C,N]
    warped_concat4 = gathered_flat.view(B, 4, BC, H, W)  # [B,4,3+C,H,W]

    # in-bounds mask from oob
    ob_mask4 = oob.unsqueeze(2)                    # [B,4,1,H,W]

    # Split back into world positions / features
    warped_wpos4 = warped_concat4[:, :, :3]        # [B,4,3,H,W]
    warped_x4    = warped_concat4[:, :, 3:]        # [B,4,C,H,W]

    # --------------------------------------------------------------------------
    # 4. DEPTH + DISOCCLUSION TEST (per tap)
    # --------------------------------------------------------------------------
    cam_pos = cam['position'].view(B, 3, 1, 1)
    cam_fwd = (cam['target'] - cam['position']).view(B, 3, 1, 1)
    cam_fwd = cam_fwd / cam_fwd.norm(dim=1, keepdim=True).clamp(min=1e-8)

    depth_curr = ((wpos - cam_pos) * cam_fwd).sum(dim=1, keepdim=True)  # [B,1,H,W]

    cam_pos4 = cam_pos.unsqueeze(1)  # [B,1,3,1,1]
    cam_fwd4 = cam_fwd.unsqueeze(1)  # [B,1,3,1,1]

    depth_prev4 = ((warped_wpos4 - cam_pos4) * cam_fwd4).sum(dim=2, keepdim=True)  # [B,4,1,H,W]

    ratio4 = depth_prev4 / depth_curr.clamp(min=rel_eps).unsqueeze(1)  # [B,4,1,H,W]
    valid4 = torch.isfinite(ratio4)

    # env hits
    env_prev4 = (warped_wpos4 == 0).all(dim=2, keepdim=True)     # [B,4,1,H,W]
    env_curr  = (wpos == 0).all(dim=1, keepdim=True)             # [B,1,H,W]
    env_new4  = env_curr.unsqueeze(1) & ~env_prev4               # [B,4,1,H,W]

    big_err4 = (ratio4 - 1).abs() > rel_threshold

    disocc4 = (ob_mask4 | env_new4 | ~valid4 | big_err4).squeeze(2)

    # --------------------------------------------------------------------------
    # 5. MASKED BILINEAR RECONSTRUCTION
    # --------------------------------------------------------------------------
    # Bilinear weights for the same TL,TR,BL,BR order:
    #   0: TL => (1-fx)*(1-fy)
    #   1: TR => fx*(1-fy)
    #   2: BL => (1-fx)*fy
    #   3: BR => fx*fy
    weights = torch.stack([
        (1 - fx) * (1 - fy),   # TL
        fx       * (1 - fy),   # TR
        (1 - fx) * fy,         # BL
        fx       * fy,         # BR
    ], dim=1)                  # [B,4,H,W]
    weights = weights.unsqueeze(2)  # [B,4,1,H,W]

    # Mask out disoccluded taps
    valid_mask4 = (~disocc4).unsqueeze(2)         # [B,4,1,H,W], bool
    w_masked    = weights * valid_mask4

    # Normalization
    norm = w_masked.sum(dim=1)                    # [B,1,H,W]

    # Weighted sum over taps
    weighted = (w_masked * warped_x4).sum(dim=1)  # [B,C,H,W]

    eps = torch.finfo(x.dtype).smallest_normal
    out = weighted / (norm + eps)                 # [B,C,H,W]

    return out, norm.to(torch.float16)


def get_remaining_time_seconds():
    """Seconds left in the current Slurm job, or None when not running under Slurm."""
    end_ts = os.environ.get("SLURM_JOB_END_TIME")
    if end_ts is None:
        return None
    return max(0, int(end_ts) - int(time.time()))


def mape(
    I: torch.Tensor,
    R: torch.Tensor,
    *,
    eps_scale: float = 0.01,
    reduction: Reduction = "none",
    channel_dim: int = -3,          # assumes ... x C x H x W by default
    keepdim: bool = True,
) -> torch.Tensor:
    """
    Mean Absolute Percentage Error-like metric.

    Computes the grayscale-denominator variant suggested for image comparison:

        gray_ref(x, y) = mean_c(R(x, y))
        eps = eps_scale * mean_xy(gray_ref)
        mape(x, y) = mean_c( |I(x, y) - R(x, y)| / (gray_ref(x, y) + eps) )

    Conventions:
      - expects channels-first by default: (..., C, H, W)
      - supports arbitrary leading batch dims
      - reduction:
          - "none": returns (..., 1, H, W) if keepdim else (..., H, W)
          - "mean": scalar
          - "sum":  scalar
    """
    if I.shape != R.shape:
        raise ValueError(f"I and R must have the same shape, got {I.shape} vs {R.shape}")

    # Move the chosen channel dim to canonical -3 so we can reduce reliably
    if channel_dim != -3:
        I = I.movedim(channel_dim, -3)
        R = R.movedim(channel_dim, -3)

    # Per-pixel grayscale reference: (..., 1, H, W)
    gray_ref = R.mean(dim=-3, keepdim=True)

    # One scalar epsilon per sample, based on the mean grayscale level.
    # This matches:
    #   denom = mean_c(R_xy) + eps_scale * mean_xyc(R_xyc)
    gray_mean = gray_ref.mean(dim=(-2, -1), keepdim=True)
    eps = torch.as_tensor(eps_scale, dtype=R.dtype, device=R.device) * gray_mean

    denom = (gray_ref + eps).clamp_min(torch.finfo(R.dtype).eps)
    per_channel = (I - R).abs() / denom

    # mean over channels: (..., 1, H, W) if keepdim else (..., H, W)
    out = per_channel.mean(dim=-3, keepdim=keepdim)

    if reduction == "none":
        return out
    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    raise ValueError(f"Invalid reduction: {reduction!r}")


def select_variants_from_pickles(
    out_dir: Path,
    allowed_variants: List[str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recursively enumerate snapshot pickles under `out_dir` and group them by variant.

    Variant deduction:
      - If filename has suffix: network-snapshot-<kimg>-<suffix>.pkl -> variant = f"ema-{suffix}"
      - If no suffix: network-snapshot-<kimg>.pkl              -> variant = "ema"  (single bucket)

    Returns:
      {
        "ema-0.025": [ {snapshot, kimg, cur_nimg, meta}, ... ],
        "ema-0.050": [ ... ],
        "ema":       [ ... ],   # only if suffixless snapshots exist
      }

    Records are sorted by:
      1) kimg ascending
      2) snapshot path string ascending (stable tie-break)
    """

    _SNAPSHOT_RE = re.compile(
        r"^network-snapshot-(?P<kimg>\d+)(?:-(?P<suffix>.+?))?\.pkl$"
    )

    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing out dir: {out_dir}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Recursive search
    for p in out_dir.rglob("network-snapshot-*.pkl"):
        if not p.is_file():
            continue

        m = _SNAPSHOT_RE.match(p.name)
        if not m:
            # If you want to be strict, you can raise here; for now we just skip non-matching.
            continue

        kimg_int = int(m.group("kimg"))
        suffix = m.group("suffix")

        variant = f"ema-{suffix}" if suffix is not None else "ema"

        if allowed_variants is not None and variant not in allowed_variants:
            continue

        rec = {
            # Keep your existing downstream expectation: `snapshot_name` is joined with run_path/"out"/snapshot_name.
            # That breaks if the snapshot is in a subdir, so we store the *relative path from out_dir*.
            "snapshot": str(p.relative_to(out_dir)).replace("\\", "/"),
            "kimg": kimg_int,
        }
        grouped[variant].append(EasyDict(rec))

    # Sort deterministically
    for v in list(grouped.keys()):
        grouped[v].sort(key=lambda r: (r["kimg"], r["snapshot"]))

    if not grouped:
        raise RuntimeError(
            f"No snapshots found under {out_dir} matching "
            f"'network-snapshot-<kimg>(-<suffix>).pkl'"
        )

    return dict(grouped)


def balanced_iter(iterable):
    seq = list(iterable)
    n = len(seq)
    if n == 0:
        return iter(())

    q = deque([(0, n - 1)])
    while q:
        lo, hi = q.popleft()
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        yield seq[mid]
        # enqueue children so we *interleave* halves (breadth-first), not exhaust left then right
        q.append((lo, mid - 1))
        q.append((mid + 1, hi))


def load_model(path):
    with dnnlib.util.open_url(str(path), verbose=False) as f:
        return pickle.load(f)['ema'].to('cuda').eval()


def to_fp32(x):
    x = to_dtype(x, torch.float16, torch.float32)
    x = to_dtype(x, torch.bfloat16, torch.float32)
    return x
