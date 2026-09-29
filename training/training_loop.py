# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified for recurrent denoising: sequence datasets, truncated
# backpropagation through time, and per-buffer whitening statistics.

"""Main training loop."""

import copy
import os
import pickle
import time
import numpy as np
import psutil
import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
import dnnlib
from dnnlib.util import EasyDict
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training import networks_edm2
from training.augmentation import ChannelBrightnessAugment, ChannelShuffleAugment, SequenceAugment, HorizontalFlipAugment

#----------------------------------------------------------------------------
# Augmentations. Radiance-like buffers get a shared random brightness scale
# and RGB permutation; all buffers are flipped horizontally together.

SHUFFLE_BUFFERS = ['indirect_f', 'target', 'restir_correlated', 'restir_correlated_srr40', 'restir_correlated_srr80', 'restir_checkerboard', 'restir_uncorrelated', 'restir_uncorrelated_reprojected', 'diffuse', 'reproj_blurred']
SCALE_BUFFERS   = ['indirect_f', 'target', 'restir_correlated', 'restir_correlated_srr40', 'restir_correlated_srr80', 'restir_checkerboard', 'restir_uncorrelated', 'restir_uncorrelated_reprojected', 'reproj_blurred']

def make_augs():
    return [
        ChannelBrightnessAugment(SCALE_BUFFERS),
        ChannelShuffleAugment(SHUFFLE_BUFFERS),
        HorizontalFlipAugment(p=0.5),
    ]

#----------------------------------------------------------------------------
# Helpers that catch exceptions in SummaryWriter (related to NaNs etc.)

def add_histogram(writer: SummaryWriter, tag, values, global_step, walltime):
    try:
        writer.add_histogram(tag, values, global_step=global_step, walltime=walltime)
    except Exception as e:
        print('SummaryWriter::add_histogram exception:', e)

def add_scalar(writer: SummaryWriter, tag, scalar_value, global_step, walltime):
    try:
        writer.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)
    except Exception as e:
        print('SummaryWriter::add_scalar exception:', e)

class DummyScaler:
    """A no-op GradScaler replacement when AMP is disabled."""
    def scale(self, loss):
        return loss
    def step(self, optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)
    def update(self, *args, **kwargs):
        pass
    def unscale_(self, optimizer):
        pass

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=0):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------

def to_fp32(x):
    x = misc.to_dtype(x, torch.float16, torch.float32)
    x = misc.to_dtype(x, torch.bfloat16, torch.float32)
    return x

def compute_mu_and_sigma(dataset, batch_gpu, data_loader_kwargs, device, augment=None, buffer_transforms=None, n=2<<14):
    """Channelwise mean and std of every (transformed, augmented) buffer, reduced over all ranks."""
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=0,
        start_idx=0,
    )
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(
        dataset=dataset,
        sampler=dataset_sampler,
        batch_size=batch_gpu,
        **data_loader_kwargs
    ))

    sum_map = {}
    sqsum_map = {}
    count_map = {}

    for _ in range(n // batch_gpu):
        x = next(dataset_iterator)
        x = misc.move_to_device(x, device)
        x = to_fp32(x)

        augment.reseed()
        x = augment(x)
        x = misc.transform(x, buffer_transforms)

        for name, tensor in x['buffers'].items():
            tensor = tensor.float()
            N, C = tensor.shape[:2]
            values = tensor.view(N, C, -1).permute(1, 0, 2).reshape(C, -1)
            count = values.shape[1]
            if count == 0:
                continue

            sum_vals = values.sum(dim=1)
            sqsum_vals = (values ** 2).sum(dim=1)
            if name not in sum_map:
                sum_map[name] = sum_vals.clone()
                sqsum_map[name] = sqsum_vals.clone()
                count_map[name] = count
            else:
                sum_map[name] += sum_vals
                sqsum_map[name] += sqsum_vals
                count_map[name] += count

    # All-reduce across all processes
    for name in sum_map:
        for t in [sum_map[name], sqsum_map[name]]:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        count_tensor = torch.tensor(float(count_map[name]), device=device)
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
        count_map[name] = count_tensor.item()

    mu_map = {}
    sigma_map = {}
    for name in sum_map:
        mean = sum_map[name] / count_map[name]
        var = (sqsum_map[name] / count_map[name]) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        mu_map[name] = mean.reshape(1, -1, 1, 1)
        sigma_map[name] = std.reshape(1, -1, 1, 1)

    return mu_map, sigma_map

def get_ref_batch(dataset):
    return torch.utils.data._utils.collate.default_collate([dataset[0]])

# Hooks for misc.TBPTT; also used by validate.py.
def _preprocess_sample(x, state):
    x = to_fp32(x)
    x = state.augment(x)
    return x

def _on_new_sequence(state):
    state.augment.reseed() # Reseed augmentations
    state.h = None # Reset history

def _save_snapshot(*, run_dir, fname, net_to_save, train_ds_kwargs, loss_fn):
    """Save a network snapshot to run_dir/fname."""
    fullpath = os.path.join(run_dir, fname)
    os.makedirs(os.path.dirname(fullpath) or run_dir, exist_ok=True)

    data = dnnlib.EasyDict(dataset_kwargs=train_ds_kwargs, loss_fn=loss_fn)
    data.ema = copy.deepcopy(net_to_save).cpu().eval().requires_grad_(False).to(torch.float16)

    dist.print0(f"Saving {fname} ... ", end="", flush=True)
    with open(fullpath, "wb") as f:
        pickle.dump(data, f)
    dist.print0("done")

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    train_ds_kwargs     = None,     # List of dataset kwargs; the datasets are concatenated.
    network_kwargs      = None,
    loss_kwargs         = dict(class_name='training.loss.RecurrentL1Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule', rampup_Mimg=0.010),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA', stds=[0.001, 0.01, 0.025, 0.050]),

    tbptt_step          = 1,        # Frames to advance between truncation boundaries.
    tbptt_window        = 1,        # Frames per backpropagation window.
    use_history         = True,     # Feed the recurrent state to the network?
    use_gradscaler      = False,    # Use torch.cuda.amp.GradScaler (for float16 networks).
    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    num_workers         = 8,        # Data loader workers per accumulation round.
    total_nimg          = 8<<30,    # Train for a total of N training frames.
    slice_nimg          = None,     # Train for a maximum of N training frames in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training frames. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training frames. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training frames. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    # Setup dataset.
    dist.print0('Loading datasets...')
    train_ds = torch.utils.data.ConcatDataset([dnnlib.util.construct_class_by_name(**kw) for kw in train_ds_kwargs])
    nframes = train_ds.datasets[0].get_nframes()
    ref_batch = get_ref_batch(train_ds)

    # Whitening statistics, only for buffers that are passed to the network.
    dist.print0('Computing buffer statistics...')
    sigma_mu_augment = SequenceAugment(seed=dist.get_rank(), augs=make_augs())
    mu_map, sigma_map = compute_mu_and_sigma(train_ds, 16, data_loader_kwargs, device, augment=sigma_mu_augment, buffer_transforms=network_kwargs['buffer_transforms'], n=4096)
    input_buffer_set = set(network_kwargs['input_buffers'] + ['target'])
    mu_map = {k: v for k, v in mu_map.items() if k in input_buffer_set}
    sigma_map = {k: v for k, v in sigma_map.items() if k in input_buffer_set}

    # Construct network.
    dist.print0('Constructing network...')
    def get_buffer(buffer):
        if buffer in network_kwargs['buffer_transforms']:
            return network_kwargs['buffer_transforms'][buffer](ref_batch['buffers'][buffer])
        return ref_batch['buffers'][buffer]
    img_channels = sum([get_buffer(buffer).shape[1] for buffer in network_kwargs['input_buffers']])
    network_kwargs = dict(network_kwargs, dtype=dnnlib.util.get_obj_by_name(network_kwargs['dtype']))
    interface_kwargs = dict(
        img_resolution = ref_batch['buffers']['target'].shape[-2],
        img_channels   = img_channels,
        mu_map         = mu_map,
        sigma_map      = sigma_map)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None
    scaler = torch.cuda.amp.GradScaler() if use_gradscaler else DummyScaler()

    # Print network summary.
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
        summary_batch = to_fp32(misc.move_to_device(ref_batch, device))
        misc.print_module_summary(net, [summary_batch], max_nesting=2)

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema, scaler=scaler)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg

    # One TBPTT iterator per accumulation round. Each walks through its own
    # set of sequences, carrying the recurrent state across windows.
    tbptts = []
    for i in range(num_accumulation_rounds):
        dataset_sampler = misc.RoundRobinSampler(
            nseqs=len(train_ds) // nframes,
            nframes=nframes,
            batch_size=batch_gpu_total,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            shuffle=True,
            split=num_accumulation_rounds,
            split_idx=i,
            start_idx=state.cur_nimg // batch_size // num_accumulation_rounds,
            seed=0)
        train_loader = dnnlib.util.construct_class_by_name(dataset=train_ds, batch_sampler=dataset_sampler, **data_loader_kwargs)
        tbptts.append(misc.TBPTT(
            step_size=tbptt_step,
            window_len=tbptt_window,
            device=device,
            iterator=iter(train_loader),
            preprocess=_preprocess_sample,
            on_new_sequence=_on_new_sequence,
            state=EasyDict(
                augment=SequenceAugment(seed=i * dist.get_world_size() + dist.get_rank(), augs=make_augs()),
                h=None,
                device=device
            )
        ))

    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    networks_edm2._mag_checker_squared_exp.clear()  # only rank0 has data, sync will fail
    networks_edm2._mean_checker_exp.clear()  # only rank0 has data, sync will fail

    latest_frame = 0
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            tbargs = dict(global_step=state.cur_nimg//1000, walltime=cur_time)
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

                stats = training_stats.default_collector.as_dict()
                for tag in ['Loss/loss', 'Loss/learning_rate', 'Grad/total_norm']:
                    if tag in stats:
                        writer.add_scalar(tag, stats[tag]['mean'], stats['Progress/kimg']['mean'])

            # Report activation magnitudes
            reduced_means = {}  # convs: one value per channel
            for k, v in networks_edm2._mag_checker_squared_exp.items():
                x_mean = v.clone().to(device)
                torch.distributed.all_reduce(x_mean, op=torch.distributed.ReduceOp.SUM)
                x_mean /= dist.get_world_size()
                reduced_means[k] = x_mean.sqrt().cpu().numpy()
            if dist.get_rank() == 0:
                for k, v in reduced_means.items():
                    if v.size > 1:
                        add_scalar(writer, f'Magnitudes/{k}_{v.size}ch_mean', v.mean(), **tbargs)
                    else:
                        add_scalar(writer, f'Magnitudes/{k}', v.mean(), **tbargs)
                if reduced_means:
                    add_histogram(writer, 'Hist/Mag/Encoder', np.array([v.mean() for k, v in reduced_means.items() if 'enc/' in k]), **tbargs)
                    add_histogram(writer, 'Hist/Mag/Decoder', np.array([v.mean() for k, v in reduced_means.items() if 'dec/' in k]), **tbargs)

            # Report activation means: one scalar per key
            reduced_means = {}
            for k, v in networks_edm2._mean_checker_exp.items():
                if v is None:
                    continue
                x = v.clone().to(device)
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
                x /= dist.get_world_size()
                reduced_means[k] = float(x.mean().item())
            if dist.get_rank() == 0:
                for k, scalar in reduced_means.items():
                    add_scalar(writer, f'Means/{k}', scalar, **tbargs)

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshots: one per EMA profile plus the raw network.
        # Validation is run offline on these snapshots (see validate.py).
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and state.cur_nimg != start_nimg:
            if dist.get_rank() == 0:
                kimg = state.cur_nimg // 1000
                ema_list = ema.get() if ema is not None else []
                variants = [(ema_net, f"network-snapshot-{kimg:07d}{suffix}.pkl") for ema_net, suffix in ema_list]
                variants.append((net, f"raw-networks/network-snapshot-{kimg:07d}-raw.pkl"))
                for net_to_save, fname in variants:
                    _save_snapshot(run_dir=run_dir, fname=fname, net_to_save=net_to_save, train_ds_kwargs=train_ds_kwargs, loss_fn=loss_fn)
            torch.distributed.barrier()

        # Save training state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Under Slurm, checkpoint and exit at a sequence boundary shortly before the job's time limit.
        remaining_time = misc.get_remaining_time_seconds()
        if remaining_time is not None and latest_frame == nframes - 1 and remaining_time < 5 * 60:
            dist.print0("Saving checkpoint and exiting...")
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)
            break

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)

        # Truncated backpropagation through time
        for tbptt in tbptts:
            with misc.ddp_sync(ddp, True):
                preds = []
                refs  = []
                for x in tbptt.step():
                    latest_frame = x['frame_idx'][0]
                    y, tbptt.state.h = ddp(x, tbptt.state.h if use_history else None)
                    preds.append(y.to(torch.float32))
                    refs.append(x)

                # Detach between truncation boundaries
                tbptt.state.h = misc.detach_tensors(tbptt.state.h)

                loss = loss_scaling * loss_fn(preds, refs) / num_accumulation_rounds
                training_stats.report('Loss/loss', loss)
                if loss != 0:
                    scaler.scale(loss).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        scaler.unscale_(optimizer)
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0, norm_type=2.0)
        training_stats.report('Grad/total_norm', float(total_norm))
        scaler.step(optimizer)
        scaler.update()

        # Update EMA and training state. Each step consumes tbptt_window frames per sequence.
        state.cur_nimg += tbptt_window * batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

#----------------------------------------------------------------------------
