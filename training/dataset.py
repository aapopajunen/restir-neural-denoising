# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming rendered frame sequences from HDF5 datasets."""

import os
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.utils.data
import h5py



class PickleCacheMixin:
    """A mixin for classes that need to cache their state using pickle."""
    def __init__(self, *args: list, **kwargs: dict) -> None:
        """Initialize the pickle path and call the parent constructor."""
        self._pickle_path = None
        super().__init__(*args, **kwargs)  # cooperative multiple inheritance

    def presave_pickle(self, pickle_path: str) -> None:
        """Save the full object state to a pickle file and store the pickle path."""
        self._pickle_path = pickle_path
        with open(pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __getstate__(self) -> dict:
        """If presaved, only pickle reference. Otherwise pickle everything."""
        if self._pickle_path:
            return {"_pickle_path": self._pickle_path}
        else:
            return self.__dict__

    def __setstate__(self, state: dict) -> None:
        """Restore object from pickle, reloading from presaved file if possible."""
        self.__dict__.update(state)
        if self._pickle_path and os.path.exists(self._pickle_path):
            with open(self._pickle_path, "rb") as f:
                saved_state = pickle.load(f) # noqa: S301
                self.__dict__.update(saved_state)



#----------------------------------------------------------------------------
# Per-scene mean target radiance. Radiance buffers are divided by this value
# so that all scenes have a comparable exposure. If the HDF5 file has a
# 'mean_radiance' attribute it is used; otherwise the scene is looked up in
# this table (the values the thesis models were trained with); otherwise 1.0.

SCENE_MEAN_RADIANCE = {
    # Applies to the -sequences, -validation and -test splits of each scene.
    'bistro-exterior':      np.float16(1.2217893),
    'bistro-interior':      np.float16(2.4126480),
    'emerald-square':       np.float16(0.7939689),
    'emerald-square-day':   np.float16(0.7939689),
    'veach-ajar':           np.float16(0.2542579),
    'zero-day':             np.float16(0.0110524),
}

RADIANCE_BUFFERS = ['target', 'restir_correlated', 'restir_correlated_srr40', 'restir_correlated_srr80', 'restir_uncorrelated', 'restir_uncorrelated_reprojected', 'reproj_blurred']

def _scene_key(path):
    """'.../165-bistro-exterior-test/dataset.hdf5' -> 'bistro-exterior'."""
    name = Path(path).parent.name
    name = name.split('-', 1)[1] if name.split('-', 1)[0].isdigit() else name
    for split in ('-sequences', '-validation', '-test'):
        if name.endswith(split):
            return name[:-len(split)]
    return name

def get_mean_radiance(path, h5_attrs):
    if 'mean_radiance' in h5_attrs:
        return np.float16(h5_attrs['mean_radiance'])
    return SCENE_MEAN_RADIANCE.get(_scene_key(path), np.float16(1.0))

#----------------------------------------------------------------------------
# Dataset of rendered frame sequences. Items are single frames; consecutive
# indices walk through one sequence, so a sampler that hands out contiguous
# index ranges yields temporally coherent batches.
#
# Expected HDF5 layout:
#   buffers         float16 [nsequences, nframes, channels, H, W]
#   cameras         structured [nsequences, nframes] with fields position,
#                   target, up, focalLength, aspectRatio, nearPlane, farPlane
#   crop_sequences  [ncrop_sizes, nsequences, nframes] of (top, left), only
#                   needed when crop_size is given
#   attrs['buffer_mapping']  repr of {buffer_name: [channel indices]}
#   attrs['crop_sizes']      repr of the list of crop sizes in crop_sequences

class MotionCompensatedDataset(PickleCacheMixin, torch.utils.data.Dataset):
    def __init__(self,
        path,                       # Path to dataset.hdf5.
        nframes          = None,    # Frames per sequence. None = full sequence length.
        buffers          = None,    # Subset of buffers to load. None = all.
        crop_size        = None,    # Square crop size. None = full frames.
        shuffle_channels = False,   # Unused; kept for compatibility with saved dataset kwargs.
        scale_brightness = False,   # Unused; kept for compatibility with saved dataset kwargs.
        fixed_crop       = False,   # Use one random crop per sequence instead of stored crop_sequences.
        split            = "train", # 'train', 'val' or 'all'.
        val_ratio        = 0.2,     # Fraction of sequences held out when split != 'all'.
    ):
        self.path = path
        self.fixed_crop = fixed_crop

        with h5py.File(path, 'r') as f:
            self.mean_radiance   = get_mean_radiance(path, f.attrs)
            self.raw_shape       = f['buffers'].shape  # [N, T, C, H, W]
            self.full_resolution = self.raw_shape[-2:] # H x W
            self.buffer_mapping  = eval(f.attrs['buffer_mapping'])
            self.all_buffers     = list(self.buffer_mapping.keys())

            if crop_size is not None:
                crop_sizes = eval(f.attrs['crop_sizes'])
                self.crop_sequences = f['crop_sequences'][crop_sizes.index(crop_size),...]
                self.crop_size = (crop_size, crop_size)
                self.infinite  = True
            else:
                self.crop_size = self.full_resolution
                self.infinite  = False
                self.crop_sequences = np.zeros(self.raw_shape[:2], dtype=[('top', 'i4'), ('left', 'i4')])

        if buffers is not None:
            assert all([buffer in self.all_buffers for buffer in buffers])
            self.all_buffers = buffers

        self.nframes = self.crop_sequences.shape[1] if nframes is None else nframes
        assert self.nframes <= self.crop_sequences.shape[1], "nframes cannot exceed the raw sequence length"

        # Split at sequence level.
        all_samples = np.arange(self.crop_sequences.shape[0])
        np.random.RandomState(0).shuffle(all_samples)
        if split != "all":
            val_size = int(len(all_samples) * val_ratio)
            self.valid_samples = list(all_samples[val_size:]) if split == "train" else list(all_samples[:val_size])
        else:
            self.valid_samples = all_samples
        self.nsamples = len(self.valid_samples)

        self.close_file() # Close files before pickling
        self.presave_pickle(f'{os.path.splitext(path)[0]}_{split}.pkl')

    def __len__(self):
        # Cropped training data is sampled indefinitely with fresh start frames and crops.
        if self.infinite:
            return self.nframes * (2**30 // self.nframes)
        return len(self.valid_samples) * self.nframes

    def get_file(self):
        if getattr(self, "file", None) is None:
            self.file = h5py.File(self.path, mode='r', libver='latest')
        return self.file

    def close_file(self):
        if getattr(self, "file", None) is not None:
            try:
                self.file.close()
            except Exception:
                pass
        self.file = None

    def get_separate_buffers(self, sample_idx, frame_idx, crop_pos):
        top, left = crop_pos
        crop_h, crop_w = self.crop_size
        data = self.get_file()['buffers'][sample_idx, frame_idx, :, top:top + crop_h, left:left + crop_w]

        sample = { 'buffers': {} }
        for buffer in self.all_buffers:
            if buffer == 'restir_checkerboard': # Skipped, as in the thesis runs.
                continue
            channels = self.buffer_mapping[buffer]
            if buffer in RADIANCE_BUFFERS:
                sample['buffers'][buffer] = data[channels,...] / self.mean_radiance
            else:
                sample['buffers'][buffer] = data[channels,...]

        sample['camera'] = self.to_camera_dict(self.get_file()['cameras'][sample_idx][frame_idx])
        return sample

    def to_camera_dict(self, cam_rec):
        return {
            'position': cam_rec['position'],
            'target': cam_rec['target'],
            'up': cam_rec['up'],
            'focalLength': cam_rec['focalLength'].item(),
            'aspectRatio': cam_rec['aspectRatio'].item(),
            'nearPlane': cam_rec['nearPlane'].item(),
            'farPlane': cam_rec['farPlane'].item(),
        }

    def __getitem__(self, idx):
        epoch = idx // (len(self.valid_samples) * self.nframes)
        sample_idx = self.valid_samples[(idx // self.nframes) % len(self.valid_samples)]

        # Random start frame (and crop, if fixed_crop) per sequence and epoch.
        rng = np.random.RandomState(hash((epoch, sample_idx)) % (1 << 31))
        start_frame = rng.randint(0, self.raw_shape[1] - self.nframes + 1)
        frame_idx = start_frame + idx % self.nframes

        H,W = self.full_resolution
        if self.fixed_crop:
            top, left = rng.randint(0, H - self.crop_size[0]), rng.randint(0, W - self.crop_size[0])
        else:
            top, left = self.crop_sequences[sample_idx, frame_idx]

        sample = self.get_separate_buffers(sample_idx % self.raw_shape[0], frame_idx, (top, left))

        # Motion vectors are stored in UV units; convert to pixels (ordered dx, dy).
        if 'motion_vector' in sample['buffers']:
            sample['buffers']['motion_vector'] *= np.array([W, H])[:,None,None]

        sample['crop_offset'] = np.array([top, left])
        sample['frame_idx'] = frame_idx - start_frame
        sample['full_resolution'] = np.array(self.full_resolution)
        return sample

    def get_crop_size(self):
        return self.crop_size

    def get_nframes(self):
        return self.nframes

    def get_nsamples(self):
        return self.nsamples

    def __del__(self):
        self.close_file()

    @property
    def name(self):
        return self.path

    @property
    def resolution(self):
        return self.raw_shape[-1]

#----------------------------------------------------------------------------
