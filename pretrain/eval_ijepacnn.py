
# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from pretrain.train_ijepacnn import IJEPA_CNN
from data.hdf5_iqfmfolder import IQTransformations, ExclusiveComposeTransforms, HDF5IQDataset, ChannelDropping, ChannelMasking
from torch.utils.data import DataLoader, Subset
import argparse
from lightly.data import LightlyDataset
import numpy as np
from sklearn.model_selection import train_test_split

def make_train_val_datasets(
    base_ds,
    transform,
    val_transform,
    mode="percentage",       # "percentage" or "per_class"
    test_size=0.3,
    per_class_n=500
):
    """
    mode="percentage" -> stratified train/test split by percentage
    mode="per_class"  -> take `per_class_n` samples per class into train
    """

    # HDF5IQDataset already keeps labels in memory. Avoid iterating through
    # every IQ sample just to construct stratification labels for large files.
    if hasattr(base_ds, "labels"):
        labels = base_ds.labels.detach().cpu().numpy()
    else:
        labels = np.array([y for _, y in base_ds])
    all_idx = np.arange(len(base_ds))

    if mode == "percentage":
        # ---- Percentage split ----
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=test_size,
            stratify=labels
        )

    elif mode == "per_class":
        # ---- Fixed samples per class ----
        classes = np.unique(labels)
        train_idx = []

        for c in classes:
            c_idx = np.where(labels == c)[0]
            if len(c_idx) < per_class_n:
                print(f"[warn] Class {c} has only {len(c_idx)} samples; taking all.")
                chosen = c_idx
            else:
                chosen = np.random.choice(c_idx, size=per_class_n, replace=False)
            train_idx.append(chosen)

        train_idx = np.concatenate(train_idx)
        np.random.shuffle(train_idx)

        val_mask = np.ones(len(base_ds), dtype=bool)
        val_mask[train_idx] = False
        val_idx = all_idx[val_mask]

    else:
        raise ValueError("mode must be 'percentage' or 'per_class'")

    # Wrap into LightlyDataset
    train_dataset = LightlyDataset.from_torch_dataset(Subset(base_ds, train_idx), transform=transform)
    train_dataset.datasets = LightlyDataset.from_torch_dataset(Subset(base_ds, train_idx), transform=transform)

    val_dataset   = LightlyDataset.from_torch_dataset(Subset(base_ds, val_idx),   transform=val_transform)

    return train_dataset, val_dataset

# ... imports stay the same ...

def main():
    parser = argparse.ArgumentParser(description="Evaluate IJEPA_CNN from checkpoint on a dataset.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--task', type=str, choices=['rml', 'rf', 'deepbeam', 'aoa', 'mod','jamming','tpi','inter',"radar","craw"],
                        help='Predefined dataset task (rml, rf, deepbeam, aoa, mod)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Explicit downstream HDF5 path; overrides the legacy task mapping.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # NEW: choose per-class sample size; batch size is derived from this
    # Suggested mapping:
    #   n_sample:  1,  10,  50, 100, 200, 500
    #   batch:     2,   4,   8,  16,  32,  64
    parser.add_argument('--sample', type=int, default=500,
                        choices=[1, 10, 50, 100, 200, 500],
                        help='Number of training samples per class (per_class_n); '
                             'batch size is derived automatically.')

    args = parser.parse_args()
    seed_all(args.seed)

    # Map n_sample -> batch_size
    #sample2batch = {1: 2, 10: 4, 50: 8, 100: 16, 200: 32, 500: 64}
    sample2batch = {1: 2, 10: 2, 50: 4, 100: 8, 200: 8, 500: 16}
    train_batch_size = sample2batch[args.sample]
    val_batch_size = 256  # keep eval larger/stable
    print(f"[Eval] Using per_class_n (n_sample) = {args.sample}")
    print(f"[Eval] Derived train batch size    = {train_batch_size}")
    print(f"[Eval] Validation batch size       = {val_batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # ----- data path selection (unchanged) -----
    if args.task is not None:
        if args.data_path is not None:
            data_path = args.data_path
        elif args.task == 'rml':
            data_path = './data/rml2016.h5'
        elif args.task == 'rf':
            data_path = './data/RF_FingerPrinting.h5'
            
        
        elif args.task == 'tpi':
            data_path = './data/tpi.h5'
        
        elif args.task == 'jamming':
            data_path = './data/jamming.h5'
        
        
        elif args.task == 'inter':
            data_path = './data/inter.h5'

        elif args.task == 'radar':
            data_path = './data/radar.h5'


        elif args.task == 'craw':
            data_path = './data/craw.h5'

        
        
        elif args.task == 'deepbeam':
            data_path = './data/deepbeam.h5'
        elif args.task in ('aoa', 'mod'):
            data_path = './data/iqfm-val.h5'
        else:
            raise ValueError(f"Unknown task: {args.task}")
    else:
        raise ValueError("You must provide --task.")

    # ----- model load (unchanged) -----
    print(f"Loading model from {args.checkpoint}")
    model = IJEPA_CNN.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    # ----- transforms (no-op example, unchanged logic) -----
    transform = ExclusiveComposeTransforms([])
    val_transform = ExclusiveComposeTransforms([])

    # ----- dataset -----
    base_ds = HDF5IQDataset(data_path, task=args.task, inter_channel=False)

    # (If you keep this line due to previous behavior)
    dataset = LightlyDataset.from_torch_dataset(base_ds, transform=transform)  # NOTE: historically required
 #----- split -----
    train_dataset, val_dataset = make_train_val_datasets(
        base_ds,
        transform=transform,
        val_transform=val_transform,
        mode="per_class",
        per_class_n=args.sample  # ← tie to --sample
    )

    # ----- dataloaders (train uses derived batch size) -----
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers
    )

    # ----- benchmark -----
    from pretrain.online_classification_benchmark import OnlineLinearClassificationBenckmark
    online_bench = OnlineLinearClassificationBenckmark(
        backbone=model.backbone_for_online_eval,
        num_classes=base_ds.num_classes,
        dataset_class=type(base_ds),
        train_dataset_kwargs={},
        val_dataset_kwargs={},
        input_size=256,
        batch_size=train_batch_size,     # ← keep in sync with train loader
        num_workers=num_workers,
        topk=(1, 2),
        dist_world_size=1,
        dist_rank=0,
    )

    def identity(x): return x

    print(f"Running online linear/knn benchmarks on {data_path} "
          f"with per_class_n={args.sample}, train_batch_size={train_batch_size}")
    results = online_bench.run_benchmarks_custom(
        device=device,
        dist_all_gather_fcn=identity,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_val_transform=transform,
        num_epochs=100,
    )
    print("Benchmark results:", results)

def seed_all(seed: int):
    import random
    import numpy as np
    import pytorch_lightning as pl

    random.seed(seed)
    np.random.seed(seed)

    pl.seed_everything(seed, workers=True)

    # PyTorch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    # Ensure deterministic behavior
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # For dataloader workers (when using num_workers > 0)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # NOTE: pretraining with timeshift_max=60, amplitude_scale=(0.8, 1.2), phase_shift_max=0, apply_prob=0.9
    #PYTHONPATH=. python pretrain/eval_ijepacnn.py --task deepbeam --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskmulti-block_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_0/epoch=100-step=306838.ckpt


    # NOTE: pretrain with no augmentations (only masking)
    # PYTHONPATH=. python pretrain/eval_ijepacnn.py --task deepbeam --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskmulti-block_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_2/last.ckpt

    print("Running on:", os.environ.get("HOSTNAME", "docker"), flush=True)
    os.system("nvidia-smi")
    print(torch.cuda.device_count(), "GPUs available", flush=True)
    main()
