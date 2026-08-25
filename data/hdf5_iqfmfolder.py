# datasets/iq_hdf5.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

IQ_NORMALIZE = {'mean': [7.117062341421843e-05, -0.00011567150795599446], 'std': [0.22032971680164337, 0.22032971680164337]}

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

        # (antenna, time, iq) -> (N, C, H, W) with H=antenna, W=time
        x = x.permute(1, 0, 2)   # (2, 4, 256)

        return x

class HDF5IQDataset(Dataset):
    def __init__(self, root, transform=None, task="aoa", inter_channel=False):
        self.h5_file = h5py.File(root, 'r')
        self.iq_data = self.h5_file['iq_data']  # Shape: (num_samples, 4, 2, 512)
        
        self.labels = self.h5_file['angles']    # AoA labels
        self.transform = transform

        self.inter_channel = inter_channel

         # Convert modulation labels to integer class indices
        self.modulation_labels = np.array(self.h5_file["modulation"][:], dtype=str)  # Extract modulation labels as strings
        
        self.angles = np.array(self.h5_file["angles"][:],dtype=str) 

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


    def __len__(self):
        return len(self.iq_data)

    def __getitem__(self, idx):
        # Convert IQ data and label to PyTorch tensors
        iq_sample = torch.tensor(self.iq_data[idx], dtype=torch.float32)
        label = self.labels[idx]
        
        # Apply transformations if defined
        if self.transform:
            # print("Applying following transforms:", self.transform.transforms)
            iq_sample = self.transform(iq_sample)

        # fast nearest-neighbor upsample along height (antenna)
        if self.inter_channel:
            # and cut/reshape to 256x256
            iq_sample = iq_sample.repeat(1, 64, 1)   # (2, 256, 256)
        else:
            
            iq_sample = iq_sample.repeat_interleave(64, dim=1)            # (2, 128, 256)

        return iq_sample, label


if __name__ == "__main__":
    h5_path = './data/iqfm-val.h5'
    ds = HDF5IQDataset(h5_path, transform=None, task="aoa")
    print("Num samples:", len(ds))
    print("Num classes:", len(ds.classes))
    print("First 5 classes:", ds.classes[:5])
