# datasets/iq_hdf5.py

import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

IQ_NORMALIZE = {'mean': [7.117062341421843e-05, -0.00011567150795599446], 'std': [0.22032971680164337, 0.22032971680164337]}


def upsample_antenna_axis(iq_sample, target_height=None):
    """Nearest-neighbor upsample ``(I/Q, antenna, time)`` to a square grid."""
    if iq_sample.ndim != 3:
        raise ValueError(f"Expected a 3D (I/Q, antenna, time) tensor, got {tuple(iq_sample.shape)}")

    _, num_antennas, num_time_samples = iq_sample.shape
    target_height = num_time_samples if target_height is None else target_height
    if target_height <= 0:
        raise ValueError(f"target_height must be positive, got {target_height}")

    if target_height % num_antennas == 0:
        return iq_sample.repeat_interleave(target_height // num_antennas, dim=1)

    return F.interpolate(
        iq_sample.unsqueeze(0),
        size=(target_height, num_time_samples),
        mode="nearest",
    ).squeeze(0)


# Define Augmentation Classes Old working
class IQTransformations:
    """Applies random transformations such as noise, time shifts, amplitude scaling, phase shifts, and sector swapping."""
    def __init__(self, noise_std=0.01, time_shift_max=10, amplitude_scale=(0.9, 1.1), phase_shift_max=0.1, apply_prob=0.5, swap_prob=0.9):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.amplitude_scale = amplitude_scale
        self.phase_shift_max = phase_shift_max
        self.apply_prob = apply_prob
        self.swap_prob = swap_prob  # Probability of applying sector swapping
   

    def __call__(self, iq_sample):
        if random.random() > self.apply_prob:
            return iq_sample

        # Add Gaussian noise
        iq_sample += self.noise_std * torch.randn_like(iq_sample)

        # Apply random time shift
        shift = random.randint(-self.time_shift_max, self.time_shift_max)
        iq_sample = torch.roll(iq_sample, shifts=shift, dims=-1)

        # Apply amplitude scaling
        scale = random.uniform(*self.amplitude_scale)
        iq_sample *= scale

        # Apply phase shift
        phase_shift = random.uniform(-self.phase_shift_max, self.phase_shift_max)
        iq_sample[:, 0, :] += phase_shift


        return iq_sample





class ChannelDropping:
    """Randomly drops one channel to simulate antenna failure."""
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob

    def __call__(self, iq_sample):
        # if random.random() < self.drop_prob:
        num_channels = iq_sample.shape[0]
        drop_channel = random.randint(0, num_channels - 1)
        iq_sample[drop_channel, :, :] = 0  # Drop one channel
        return iq_sample





class ChannelMasking:
    """Masks a portion of the time-domain signal to simulate interference."""
    def __init__(self, mask_prob=0.5, mask_length=10):
        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def __call__(self, iq_sample):
        # if random.random() < self.mask_prob:
        num_samples = iq_sample.shape[-1]
        start_idx = random.randint(0, num_samples - self.mask_length)
        iq_sample[:, :, start_idx:start_idx + self.mask_length] = 0
        return iq_sample

class ExclusiveComposeTransforms:
    """Combines mutually exclusive transformations into a pipeline."""
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        # Determine probabilities
        if len(self.transforms) > 2:
            apply_dropping = random.random() < self.transforms[1].drop_prob 
            apply_masking = random.random() < self.transforms[2].mask_prob
        else:
            apply_dropping = 0
            apply_masking = 0

        # Apply non-exclusive transformations
        if len(self.transforms) > 0:
            x = self.transforms[0](x)  # Apply IQTransformations
        
        # if apply_dropping:
        #     x = self.transforms[1](x)
        # if apply_masking:
        #     x = self.transforms[2](x)

        return x

class HDF5IQDataset(Dataset):
    def __init__(self, root, transform=None, task="aoa", inter_channel=False):
        self.root = os.fspath(root)
        self.h5_file = None
        self.iq_data = None
        self._h5_pid = None
        self.transform = transform

        # Retained for compatibility with older checkpoints/configs. WirelessJEPA
        # always uses nearest-neighbor antenna upsampling for every mask geometry.
        self.inter_channel = inter_channel

        # Labels are small enough to keep in memory; the large IQ dataset is
        # opened lazily and independently inside each DataLoader worker.
        with h5py.File(self.root, "r") as h5_file:
            self._length = len(h5_file["iq_data"])
            self.modulation_labels = np.array(h5_file["modulation"][:], dtype=str)
            self.angles = np.array(h5_file["angles"][:], dtype=str)

        if task == "aoa":
            print( "Selected data angle of arrival")
            self.classes = np.unique(self.angles, axis=0)

            self.label_mapping = {tuple(pair): idx for idx, pair in enumerate(self.classes)}
            self.labels = torch.tensor([self.label_mapping[tuple(mod)] for mod in self.angles], dtype=torch.long)
        else:
            print( "Selected data Modulation")

            self.classes = np.unique(self.modulation_labels)

            self.label_mapping = {mod: idx for idx, mod in enumerate(self.classes)}
            self.labels = torch.tensor([self.label_mapping[mod] for mod in self.modulation_labels], dtype=torch.long)
        
        self.num_classes = len(self.classes)

    def _ensure_open(self):
        """Open one HDF5 handle per process instead of sharing it across workers."""
        process_id = os.getpid()
        if self.h5_file is None or self._h5_pid != process_id:
            self.close()
            self.h5_file = h5py.File(self.root, "r")
            self.iq_data = self.h5_file["iq_data"]
            self._h5_pid = process_id

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
        self.h5_file = None
        self.iq_data = None
        self._h5_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["iq_data"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Convert IQ data and label to PyTorch tensors
        self._ensure_open()
        iq_sample = torch.tensor(self.iq_data[idx], dtype=torch.float32)
        label = self.labels[idx]
        
        # Apply transformations if defined
        if self.transform:
            # print("Applying following transforms:", self.transform.transforms)
            iq_sample = self.transform(iq_sample)

        # (antenna, I/Q, time) -> (I/Q, antenna, time), then apply the
        # x[c, i, t] = x_raw[c, floor(i / 64), t] mapping from the paper.
        iq_sample = iq_sample.permute(1, 0, 2)
        iq_sample = upsample_antenna_axis(iq_sample)

        return iq_sample, label


if __name__ == "__main__":
    h5_path = './data/iqfm-val.h5'
    ds = HDF5IQDataset(h5_path, transform=None, task="aoa")
    print("Num samples:", len(ds))
    print("Num classes:", len(ds.classes))
    print("First 5 classes:", ds.classes[:5])
