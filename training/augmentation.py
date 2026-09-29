

import torch

from torch_utils import misc, persistence

@persistence.persistent_class
class ChannelBrightnessAugment:
    def __init__(self, buffers, sigma=0.2):
        self.buffers = buffers
        self.sigma = sigma

    def __call__(self, g: torch.Generator, x):
        B = x['buffers']['target'].shape[0]
        device = x['buffers']['target'].device

        mu = -0.5 * (self.sigma ** 2) # Preserve brightness in expectation

        # Per-batch brightness multiplier
        bright = torch.empty(B, 1, 1, 1, dtype=torch.float32, device=device)
        bright.log_normal_(mean=mu, std=self.sigma, generator=g)

        for buf in self.buffers:
            if buf not in x['buffers']:
                continue

            t = x['buffers'][buf]
            dtype = t.dtype

            # Get dtype-dependent safe range
            finfo = torch.finfo(dtype)
            max_val = finfo.max
            min_val = finfo.min

            # Apply brightness and clamp to dtype range
            t = t * bright.to(dtype)
            t.clamp_(min_val, 1e7)
            x['buffers'][buf] = t

@persistence.persistent_class
class ChannelShuffleAugment:
    def __init__(self, buffers):
        self.buffers = buffers
        # All 6 perms of RGB – fixed lookup table
        self.perms = torch.tensor([
            [0, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ])

    def __call__(self, g: torch.Generator, x):
        B, C, H, W = x['buffers']['target'].shape
        idx = torch.randint(0, 6, (B,), generator=g, device=x['buffers']['target'].device)
        # idx: B      perms: 6×3  →  gather to B×3
        perm = self.perms.to(idx.device)[idx]        # B×3
        perm = perm.view(B, C, 1, 1).expand(-1, -1, H, W)

        for buf in self.buffers:
            if buf in x['buffers']:
                x['buffers'][buf].copy_(
                    torch.gather(x['buffers'][buf], 1, perm)
                )

@persistence.persistent_class
class HorizontalFlipAugment:
    def __init__(self, p=0.5,
                 flip_x_buffers=('normal', 'motion_vector')):
        """
        p: probability to flip each sample independently
        flip_x_buffers: buffers whose channel-0 (x) must flip sign when flipped
        """
        self.p = float(p)
        self.flip_x_buffers = tuple(flip_x_buffers)

    def __call__(self, g: torch.Generator, x):
        dev = x['buffers']['target'].device
        B, _, _, W = x['buffers']['target'].shape  # (B, C, H, W)

        # Per-sample mask: True where we flip
        flip_mask = torch.rand(B, 1, 1, 1, device=dev, generator=g) < self.p

        # Sign: +1 or -1 per sample
        sign = torch.where(flip_mask, -1.0, 1.0).view(B, 1)

        # ---- flip pixels in all buffers ----
        for name, buf in x['buffers'].items():
            flipped = buf.flip(dims=[3])  # horizontal pixel flip
            x['buffers'][name].copy_(torch.where(flip_mask, flipped, buf))

        # ---- negate x component in selected buffers ----
        for name in self.flip_x_buffers:
            if name in x['buffers']:
                x['buffers'][name][:, 0, :, :].mul_(sign.view(B, 1, 1))

        # ---- negate x component of camera fields ----
        # x['camera']['position'][:, 0].mul_(sign.view(B))
        # x['camera']['target'][:, 0].mul_(sign.view(B))
        # x['camera']['up'][:, 0].mul_(sign.view(B))

        mask = flip_mask.view(B)
        full_w = x['full_resolution'][:, 1].to(x['crop_offset'].dtype)  # match dtype
        offsets = x['crop_offset'][:, 1]                                # stays in place

        offsets[mask] = full_w[mask] - offsets[mask] - W

        return x

@persistence.persistent_class
class SequenceAugment:
    def __init__(self, augs=(), seed=0, device='cuda'):
        self.augs  = list(augs)
        self.seed  = seed
        self.device = device

    def reseed(self):
        # Same semantics as before
        self.seed = torch.randint(1 << 31, (), dtype=torch.int64).item()

    def __call__(self, x):
        g_base = torch.Generator(device=self.device)

        for aug_idx, aug in enumerate(self.augs):
            # Derive one generator per augment op, but **not** per sample
            g = g_base.manual_seed(hash((self.seed, aug_idx)) % (1 << 31))
            aug(g, x)                       # vectorised aug handles all B samples

        return misc.detach_tensors(x)

