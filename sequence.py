from pathlib import Path
from typing import Optional
import torch
from dnnlib.util import EasyDict


class Sequence:
    def __init__(self):
        self.frames = []
        self.name = ""

    def append(self, frame_data, format='bchw'):
        if format != 'bchw':
            raise ValueError("Only 'bchw' supported")

        first = next(iter(frame_data.values()))
        if first.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(first.shape)}")
        B, C, H, W = first.shape
        assert B == 1, "Batch dim should be 1"

        hwc = EasyDict()
        for k, t in frame_data.items():
            if t.ndim != 4 or t.shape[0] != B or t.shape[2:] != (H, W):
                raise ValueError(f"All tensors must share (B,H,W); '{k}' has {tuple(t.shape)}")

            # [1,C,H,W] → [H,W,C], store as CPU float16
            hwc[k] = t[0].permute(1, 2, 0).contiguous().to(torch.float16).cpu()

        self.frames.append(hwc)

    def keys(self):
        return self.frames[0].keys()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def get_tchw(self, name: str) -> torch.Tensor:
        if len(self.frames) == 0:
            raise RuntimeError("Sequence is empty")

        tensors = []
        expected_shape = None
        for frame_idx, frame in enumerate(self.frames):
            if name not in frame:
                raise KeyError(f"Buffer '{name}' not found in frame {frame_idx}")

            tensor_hwc = frame[name]
            if tensor_hwc.ndim != 3:
                raise ValueError(
                    f"Expected HWC tensor for buffer '{name}' in frame {frame_idx}, "
                    f"got shape {tuple(tensor_hwc.shape)}"
                )

            if expected_shape is None:
                expected_shape = tuple(tensor_hwc.shape)
            elif tuple(tensor_hwc.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for buffer '{name}': frame 0 has {expected_shape}, "
                    f"frame {frame_idx} has {tuple(tensor_hwc.shape)}"
                )

            tensors.append(tensor_hwc.permute(2, 0, 1).contiguous())

        return torch.stack(tensors, dim=0)

    # -----------------------------
    # Compressed save/load support
    # -----------------------------
    @staticmethod
    def _compression_from_path(path: Path) -> Optional[str]:
        """
        Returns: 'gzip' | 'bz2' | 'lzma' | None
        based on filename suffix.
        """
        suf = "".join(path.suffixes).lower()
        if suf.endswith(".gz"):
            return "gzip"
        if suf.endswith(".bz2"):
            return "bz2"
        if suf.endswith(".xz") or suf.endswith(".lzma"):
            return "lzma"
        return None

    @staticmethod
    def _open_for_torch(path: Path, mode: str):
        """
        Open a file path either as a normal binary file or wrapped in a compressor.
        mode should be 'rb' or 'wb'.
        """
        comp = Sequence._compression_from_path(path)
        if comp is None:
            return open(path, mode)

        if comp == "gzip":
            import gzip
            return gzip.open(path, mode)
        if comp == "bz2":
            import bz2
            return bz2.open(path, mode)
        if comp == "lzma":
            import lzma
            return lzma.open(path, mode)

        raise RuntimeError(f"Unknown compression type: {comp}")

    def save(self, path):
        """
        Save all frames to a single file.
        - Uncompressed: e.g. 'seq.pt'
        - Compressed (auto by extension): 'seq.pt.gz', 'seq.pt.xz', 'seq.pt.bz2'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # torch.save can write to a file-like object, so wrap as needed.
        with self._open_for_torch(path, "wb") as f:
            torch.save(self.frames, f)

    def load_(self, path):
        """
        In-place load frames from a file (compressed or not).
        Note: named load_ because a @classmethod load() also exists.
        """
        path = Path(path)
        with self._open_for_torch(path, "rb") as f:
            self.frames = torch.load(f, map_location="cpu")
        self.name = path.name
        return self

    @classmethod
    def load(cls, path):
        """Construct a Sequence from a saved file (compressed or not)."""
        obj = cls()
        obj.load_(path)
        return obj
