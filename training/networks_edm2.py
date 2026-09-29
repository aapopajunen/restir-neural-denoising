# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Recurrent denoiser built on the magnitude-preserving U-Net from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

from collections import defaultdict
from typing import Hashable
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc

# Double finite sums => should be fine to exchange order (avg of means instead of mean of averages)
_mag_checker_squared_exp = defaultdict(lambda: None)  # sqr_avg


def check_magnitude(a: torch.Tensor, key: Hashable, training: bool, ema=0.9):
    """
    Compute expected magnitude of activation tensor a.
    params:
        a: torch.Tensor of activations
        key: any hashable type, used as identifier to keep track of history
        ema: float, length of EMA decay (0=instantaneous, 1=infinite)
    """
    if not training:
        return None

    sqr_avg = _mag_checker_squared_exp[key]

    # Squaring at low precision (fp16) results in infs
    a = a.detach().float()

    batched_sqr_norms = None
    if len(a.shape) > 2:
        # Don't average across channel dim
        batched_sqr_norms = a.square().mean(dim=tuple(range(2, len(a.shape)))).cpu()  # [B, C]
    elif len(a.shape) == 2:
        batched_sqr_norms = a.square().mean(dim=1).cpu()  # (batch,)
    else:
        raise NotImplementedError('Unknown shape')

    # mean across batch dim, result is (1,) or (C,)
    sqr_norms = batched_sqr_norms.mean(dim=0)
    new_avg = sqr_norms if sqr_avg is None else (ema * sqr_avg + (1 - ema) * sqr_norms)
    _mag_checker_squared_exp[key] = new_avg

    # Individual estimate (post-sqrt) per channel
    # Return average across channels
    return new_avg.sqrt().mean(axis=0).item()


# Dictionary to track EMA of means
_mean_checker_exp = defaultdict(lambda: None)


def check_mean(a, key: Hashable, training: bool, ema: float = 0.9):
    """
    Track an exponential moving average of the mean of `a`.

    Arguments
    ---------
    a : torch.Tensor | number
        Activations or a loss value. Supports shapes:
        - 0D (scalar)
        - 1D (B,)
        - 2D (B, F) -> mean over F per sample
        - ND  (B, C, ...) -> mean over all dims >=2 (keeps channelwise stats)
    key : Hashable
        Identifier to keep separate histories.
    training : bool
        If False, do nothing and return None.
    ema : float
        EMA decay. 0 = instant (use current means), 1 = no update.

    Returns
    -------
    float | None
        The (scalar) EMA across channels for this key, or None if not training.
    """
    if not training:
        return None

    # Ensure tensor
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)

    # Detach and cast to float to avoid precision surprises (e.g., fp16)
    a = a.detach().float()

    # Compute per-sample means -> shape (B,) or (B, C)
    if a.ndim == 0:
        # scalar: treat as batch of size 1
        batched_means = a.view(1).cpu()  # (1,)
    elif a.ndim == 1:
        # (B,)
        batched_means = a.cpu()  # (B,)
    elif a.ndim == 2:
        # (B, F) -> mean over features
        batched_means = a.mean(dim=1).cpu()  # (B,)
    else:
        # (B, C, D1, D2, ...) -> mean over spatial dims; keep channels
        # Assumes channel at dim=1.
        reduce_dims = tuple(range(2, a.ndim))
        batched_means = a.mean(dim=reduce_dims).cpu()  # (B, C)

    # Mean across batch -> shape (1,) or (C,)
    means = batched_means.mean(dim=0)

    # Fetch prior EMA (CPU), update
    prev = _mean_checker_exp.get(key, None)
    new_avg = means if prev is None else (ema * prev + (1.0 - ema) * means)

    _mean_checker_exp[key] = new_avg  # stays on CPU

    # Return a scalar summary (mean over channels)
    return float(new_avg.mean().item())

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
        self.v = -1

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training and self.v != self.weight._version:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        self.v = self.weight._version
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        output_channels,
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=False, **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=False, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=False, **block_kwargs)
        self.out_conv = MPConv(cout, output_channels, kernel=[3,3])

    def forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            check_magnitude(x, f"enc/{name}", self.training)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
            check_magnitude(x, f"dec/{name}", self.training)
        return self.out_conv(x)


#----------------------------------------------------------------------------
# Recurrent denoiser used in the thesis.
#
# Each frame, the previous hidden state is reprojected into the current
# frame along the motion vectors; bilinear taps whose world-space position
# does not match (disocclusions) are dropped. The UNet sees the whitened
# input buffers together with the reprojected hidden state and predicts a
# new hidden state plus a per-pixel blend weight w. The hidden state is updated as lerp(new, reprojected, w),
# so w = 0 trusts the current frame and w = 1 reuses history. A 1x1 conv
# maps the hidden state to log radiance, which is un-whitened and mapped
# back to linear radiance.
#
# Named DPCNBlendDisocc during the thesis experiments. Snapshots from those
# runs still load, since they embed the class source they were saved with.

@persistence.persistent_class
class RecurrentDenoiser(torch.nn.Module):
    def __init__(self,
        img_resolution,             # Image resolution.
        img_channels,               # Number of input channels (sum over input_buffers).
        input_buffers,              # Names of the buffers fed to the network, in order.
        mu_map,                     # Per-buffer channelwise means, used for whitening.
        sigma_map,                  # Per-buffer channelwise std devs, used for whitening.
        dtype             = torch.bfloat16, # float16, bfloat16 or float32.
        temporal_channels = 8,      # Number of hidden state channels.
        buffer_transforms = {},     # Per-buffer transforms applied before whitening.
        **unet_kwargs,              # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution    = img_resolution
        self.img_channels      = img_channels
        self.input_buffers     = input_buffers
        self.dtype             = dtype
        self.mu_map            = self._register_dict_as_buffers("mu_map", mu_map)
        self.sigma_map         = self._register_dict_as_buffers("sigma_map", sigma_map)
        self.temporal_channels = temporal_channels
        self.buffer_transforms = buffer_transforms

        unet_cin      = img_channels + temporal_channels
        self.unet     = UNet(img_resolution=img_resolution, img_channels=unet_cin, output_channels=temporal_channels + 1, label_dim=0, **unet_kwargs)
        self.out_conv = MPConv(self.temporal_channels, 3, kernel=[1,1])

    def _register_dict_as_buffers(self, name, tensor_dict):
        out = {}
        for key, tensor in tensor_dict.items():
            buf = torch.as_tensor(tensor).detach().clone()
            buf.requires_grad_(False)
            self.register_buffer(f"{name}__{key}", buf)
            out[key] = buf
        return out

    def softclamp(self, x, limit, knee=8.0):
        return (x / (1 + (x / limit).abs().pow(knee)).pow(1.0 / knee))

    def compute_blend_weights(self, x: torch.Tensor):
        # Sigmoid-like squashing to [0, 1], evaluated in fp32.
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            w32 = torch.atan(0.25 * torch.pi * x32) / torch.pi + 0.5
        return w32.to(x.dtype)

    def whiten(self, x, buffer):
        if buffer in self.buffer_transforms:
            x = self.buffer_transforms[buffer](x)
        mu    = self.mu_map[buffer].to(x.dtype).detach().clone()
        sigma = self.sigma_map[buffer].to(x.dtype).detach().clone()
        return (x - mu) / sigma

    def unwhiten(self, x, buffer):
        mu    = self.mu_map[buffer].to(x.dtype).detach().clone()
        sigma = self.sigma_map[buffer].to(x.dtype).detach().clone()
        return x * sigma + mu

    def forward(self, x, state=None, **unet_kwargs):
        device = x['buffers']['target'].device
        B,_,H,W = x['buffers']['target'].shape

        # Previous hidden state, reprojected into the current frame.
        if state is None:
            hidden_prev = torch.zeros((B, self.temporal_channels, H, W), device=device, dtype=self.dtype)
        else:
            hidden_prev, _disocclusion = misc.masked_reproject_gather(
                x=state['hidden'],
                wpos=x['buffers']['world_space'],
                prev_wpos=state['world_space'],
                motion_vector=x['buffers']['motion_vector'],
                crop_offset=x['crop_offset'],
                prev_crop_offset=state['crop_offset'],
                cam=x['camera'],
                rel_threshold=0.01
            )
            hidden_prev = hidden_prev.to(self.dtype)

        # The UNet's noise conditioning is unused; sigma is fixed to 1 so c_noise = 0.
        sigma = torch.ones([B, 1, 1, 1], dtype=self.dtype, device=device)
        c_noise = sigma.flatten().log() / 4

        # Build UNet input from whitened buffers and the hidden state.
        sample_buffers = torch.cat([self.whiten(x['buffers'][b], b).detach() for b in self.input_buffers], dim=1)
        unet_in = torch.cat((sample_buffers.to(self.dtype), hidden_prev.to(self.dtype)), dim=1)
        unet_out = self.unet(unet_in, c_noise, class_labels=None, **unet_kwargs)

        # Split into new hidden state and blend weight.
        hidden        = unet_out[:,:self.temporal_channels,...]
        blend_weights = self.compute_blend_weights(unet_out[:,self.temporal_channels:,...])
        check_mean(blend_weights, "mean/blend_weight", self.training)

        # Temporal blend: 0 = current frame, 1 = reprojected history.
        hidden = hidden.lerp(hidden_prev, blend_weights)

        # Log radiance in fp32, soft-clamped to a reasonable range.
        out_log = self.out_conv(hidden)
        out_log = self.unwhiten(out_log.float(), 'target')
        limit = torch.log1p(torch.tensor(1e7, dtype=torch.float32, device=device))
        out_log = self.softclamp(out_log, limit)

        # Linear radiance (inverse of signed_log1p).
        out_linear = out_log.sign() * out_log.abs().expm1()

        state = {
            'crop_offset': x['crop_offset'],
            'hidden': hidden,
            'world_space': x['buffers']['world_space'],
            'blend_weights': blend_weights,
        }

        return out_linear, state

#----------------------------------------------------------------------------
