"""Run one real-IQ batch through every WirelessJEPA masking strategy."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import h5py
import hydra
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.hdf5_iqfmfolder import upsample_antenna_axis
from pretrain.iqfm_masks import WirelessMaskGenerator, masked_l2_loss


CONFIG_NAMES = (
    "ijepacnn_iqfm_random",
    "ijepacnn_iqfm_antenna",
    "ijepacnn_iqfm_time",
    "ijepacnn_iqfm_multiblock",
)


def load_config(config_name):
    config_dir = str((Path(__file__).parent / "configs").resolve())
    with hydra.initialize_config_dir(version_base="1.2", config_dir=config_dir):
        return hydra.compose(config_name=config_name)


def load_real_batch(h5_path, batch_size):
    with h5py.File(h5_path, "r") as handle:
        raw = torch.from_numpy(handle["iq_data"][:batch_size]).float()
    # (batch, antenna, I/Q, time) -> (batch, I/Q, antenna, time)
    raw = raw.permute(0, 2, 1, 3)
    return torch.stack([upsample_antenna_axis(sample) for sample in raw])


def mask_only_result(cfg, batch_size, device):
    generator = WirelessMaskGenerator(
        input_size=(256, 256),
        strategy=cfg.mask.strategy,
        patch_size=cfg.mask.patch_size,
        mask_ratio=cfg.get("mask_ratio", None),
        mask_ratio_choices=cfg.mask.get("mask_ratio_choices", None),
        multi_block_kwargs=cfg.mask.get("mutli_block_kwargs", None),
    )
    context, target = generator(batch_size, device=device)
    return {
        "strategy": str(cfg.mask.strategy),
        "patch_size": list(cfg.mask.patch_size),
        "grid": list(context.shape[-2:]),
        "context_patches": context.sum((1, 2, 3)).tolist(),
        "target_patches": target.sum((1, 2, 3)).tolist(),
    }


def full_model_result(cfg, batch):
    # Importing here lets --mask-only diagnose configs without all training packages.
    from pretrain.train_ijepacnn import IJEPA_CNN

    model = IJEPA_CNN(cfg).to(batch.device)
    model.input_size = 256
    model.setup_masking()
    model.train()
    model.zero_grad(set_to_none=True)

    prediction, _, target_mask = model(batch)
    with torch.no_grad():
        target = model.forward_momentum(batch)
    loss = masked_l2_loss(prediction, target, target_mask)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss for {cfg.mask.strategy}: {loss}")
    loss.backward()

    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError(f"Non-finite gradients for {cfg.mask.strategy}")

    result = {
        "strategy": str(cfg.mask.strategy),
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "loss": float(loss.detach().cpu()),
        "finite_gradients": finite_gradients,
    }
    del model, prediction, target, target_mask, loss
    gc.collect()
    if batch.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--train-h5", default=os.environ.get("WIRELESSJEPA_TRAIN_H5", "./data/iqfm-train.h5"))
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested, but no CUDA device is available")
    device = torch.device(device)

    batch = None if args.mask_only else load_real_batch(args.train_h5, args.batch_size).to(device)
    for config_name in CONFIG_NAMES:
        cfg = load_config(config_name)
        result = (
            mask_only_result(cfg, args.batch_size, device)
            if args.mask_only
            else full_model_result(cfg, batch)
        )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
