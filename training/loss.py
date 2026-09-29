import torch
import torch.nn as nn
from torch_utils import misc, persistence

@persistence.persistent_class
class RecurrentL1Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _stack_targets(xs, key):
        # stacks xs[t]['buffers'][key] over time -> (T, B, C, H, W) or (T, B, 2, H, W)
        return torch.stack([x['buffers'][key] for x in xs], dim=0)

    @staticmethod
    def _stack_offsets(xs):
        # stacks xs[t]['crop_offset'] over time -> (T, B, 2)
        return torch.stack([x['crop_offset'] for x in xs], dim=0)

    @staticmethod
    def _flatten_tb(x):
        # Merge (T, B, ...) -> (T*B, ...)
        return x.reshape(-1, *x.shape[2:])

    def _spatial(self, y_hats, xs):
        # Vectorized: stack over time and compute L1 over all pixels
        y_pred = torch.stack(y_hats, dim=0)                       # (T, B, C, H, W)
        y_true = self._stack_targets(xs, 'target')                # (T, B, C, H, W)

        diff = (y_pred - y_true).abs()
        return diff.mean()

    def _temporal(self, y_hats, xs):
        T = len(xs)
        if T <= 1:
            # Match original behavior: no temporal loss if fewer than 2 frames
            return torch.zeros((), device=y_hats[0].device, dtype=y_hats[0].dtype)

        # Stack predictions/targets/motion vectors over time
        y_pred  = torch.stack(y_hats, dim=0)                      # (T, B, C, H, W)
        y_true  = self._stack_targets(xs, 'target')               # (T, B, C, H, W)
        mv      = self._stack_targets(xs, 'motion_vector')        # (T, B, 2, H, W)
        offsets = self._stack_offsets(xs)                         # (T, B, 2)

        # Consecutive pairs: t-1 -> t
        y_pred_prev, y_pred_curr = y_pred[:-1], y_pred[1:]        # (T-1, B, C, H, W)
        y_true_prev, y_true_curr = y_true[:-1], y_true[1:]        # (T-1, B, C, H, W)
        mv_curr                  = mv[1:]                         # (T-1, B, 2, H, W)

        # Flow with crop offset 
        crop_diff = (offsets[:-1] - offsets[1:]).flip(-1)         # swap xy -> yx (view-like)
        crop_diff = crop_diff.unsqueeze(-1).unsqueeze(-1)         # (T-1, B, 2, 1, 1)
        flow = mv_curr - crop_diff

        # Reproject previous predictions/targets into current frame (batched over T-1 * B)
        ypp_flat   = self._flatten_tb(y_pred_prev)                # (N, C, H, W)
        ytp_flat   = self._flatten_tb(y_true_prev)                # (N, C, H, W)
        flow_flat  = self._flatten_tb(flow)                       # (N, 2, H, W)

        # --- Single reprojection by stacking channels ---
        C = ypp_flat.shape[1]
        stacked = torch.cat([ypp_flat, ytp_flat], dim=1)          # (N, 2C, H, W)
        warped  = misc.reproj(stacked, flow_flat)                 # (N, 2C, H, W)
        ypp_warped, ytp_warped = warped.split([C, C], dim=1)      # (N, C, H, W), (N, C, H, W)

        # Restore (T-1, B, C, H, W)
        ypp_warped = ypp_warped.view_as(y_pred_prev)
        ytp_warped = ytp_warped.view_as(y_true_prev)

        # Prediction difference vs. real difference
        y_hat_diff = y_pred_curr - ypp_warped
        y_diff     = y_true_curr - ytp_warped

        loss = (y_diff - y_hat_diff).abs()
        return loss.mean()

    def forward(self, y_hats, xs):
        spatial  = self._spatial(y_hats, xs)
        temporal = self._temporal(y_hats, xs)

        return spatial + temporal

