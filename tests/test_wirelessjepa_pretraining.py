import tempfile
import unittest

import h5py
import numpy as np
import torch

from data.hdf5_iqfmfolder import HDF5IQDataset, upsample_antenna_axis
from pretrain.ijepa_mask import MultiBlockMask
from pretrain.iqfm_masks import PAPER_PATCH_SIZES, WirelessMaskGenerator, masked_l2_loss


MULTI_BLOCK_KWARGS = {
    "aspect_ratio": (0.75, 1.5),
    "enc_mask_scale": (1.0, 1.0),
    "min_keep": 3,
    "num_enc_blocks": 1,
    "num_pred_blocks": 4,
    "pred_mask_scale": (0.15, 0.2),
}


class AntennaUpsamplingTests(unittest.TestCase):
    def test_nearest_neighbor_row_order(self):
        sample = torch.arange(4).view(1, 4, 1).expand(2, 4, 8)
        upsampled = upsample_antenna_axis(sample)
        self.assertEqual(tuple(upsampled.shape), (2, 8, 8))
        self.assertEqual(upsampled[0, :, 0].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_dataset_permute_and_upsample_without_transform(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as handle:
                iq = np.zeros((1, 4, 2, 8), dtype=np.float32)
                for antenna in range(4):
                    iq[0, antenna, :, :] = antenna
                handle.create_dataset("iq_data", data=iq)
                handle.create_dataset("angles", data=np.array([[0.0, 0.0]], dtype=np.float32))
                handle.create_dataset("modulation", data=np.array(["qpsk"], dtype=h5py.string_dtype()))

            dataset = HDF5IQDataset(tmp.name, transform=None)
            sample, _ = dataset[0]
            self.assertEqual(tuple(sample.shape), (2, 8, 8))
            self.assertEqual(sample[0, :, 0].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
            dataset.close()

    def test_dataset_reads_with_multiple_workers(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as handle:
                iq = np.zeros((4, 4, 2, 8), dtype=np.float32)
                for sample_index in range(4):
                    iq[sample_index] = sample_index
                handle.create_dataset("iq_data", data=iq)
                handle.create_dataset(
                    "angles",
                    data=np.array([[index, index] for index in range(4)], dtype=np.float32),
                )
                handle.create_dataset(
                    "modulation",
                    data=np.array(["qpsk"] * 4, dtype=h5py.string_dtype()),
                )

            dataset = HDF5IQDataset(tmp.name, transform=None)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
            samples, labels = next(iter(loader))
            self.assertEqual(tuple(samples.shape), (2, 2, 8, 8))
            self.assertEqual(tuple(labels.shape), (2,))
            dataset.close()


class WirelessMaskTests(unittest.TestCase):
    def make_generator(self, strategy):
        kwargs = {
            "input_size": (256, 256),
            "strategy": strategy,
            "patch_size": PAPER_PATCH_SIZES[strategy],
        }
        if strategy == "antenna":
            kwargs["mask_ratio_choices"] = (0.25, 0.5, 0.75)
        elif strategy == "multi-block":
            kwargs["mask_ratio"] = 0.6
            kwargs["multi_block_kwargs"] = MULTI_BLOCK_KWARGS
        else:
            kwargs["mask_ratio"] = 0.4 if strategy == "random" else 0.6
        return WirelessMaskGenerator(**kwargs)

    def test_paper_mask_grid_shapes_and_nonempty_targets(self):
        expected_shapes = {
            "random": (4, 8),
            "antenna": (4, 1),
            "time": (1, 8),
            "multi-block": (4, 8),
        }
        for strategy, expected_shape in expected_shapes.items():
            with self.subTest(strategy=strategy):
                generator = self.make_generator(strategy)
                context, target = generator(batch_size=4)
                self.assertEqual(tuple(context.shape), (4, 1, *expected_shape))
                self.assertEqual(tuple(target.shape), (4, 1, *expected_shape))
                self.assertTrue(torch.all(target.sum(dim=(1, 2, 3)) > 0))
                self.assertFalse(torch.any(context & target))

    def test_invalid_multiblock_fails_instead_of_hanging(self):
        mask = MultiBlockMask(
            input_size=256,
            patch_size=(256, 32),
            **MULTI_BLOCK_KWARGS,
        )
        with self.assertRaisesRegex(ValueError, "too small"):
            mask(1)

    def test_masked_l2_uses_only_target_locations(self):
        prediction = torch.ones(1, 2, 2, 2)
        target = torch.zeros_like(prediction)
        mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        mask[:, :, 0, 0] = True
        self.assertEqual(masked_l2_loss(prediction, target, mask).item(), 2.0)


if __name__ == "__main__":
    unittest.main()
