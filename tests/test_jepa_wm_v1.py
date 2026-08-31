import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.hap_uav import PilotWindowDataset, TrajectoryDataset, generate_expert_hdf5, generate_pilot_hdf5, generate_trajectory_hdf5
from environment.hap_uav import HapUavConfig, HapUavEnv, dbm_to_watts, watts_to_dbm
from evaluation.hap_uav_eval import genie_action
from models.jepa_wm import ActionConditionedMDNLSTM, JEPAWorldModelEncoder, LatentStateEncoder, PowerController, RFVAE
from training.jepa_wm import _split_pilot_dataset


class EnvironmentTests(unittest.TestCase):
    def test_power_units(self):
        self.assertAlmostEqual(float(dbm_to_watts(30.0)), 1.0, places=8)
        self.assertAlmostEqual(float(watts_to_dbm(1.0)), 30.0, places=8)

    def test_observation_and_no_hidden_channels(self):
        env = HapUavEnv(HapUavConfig(max_steps=2))
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs["iq"].shape, (2, 4, 256))
        self.assertEqual(obs["side"].shape, (8,))
        next_obs, _, _, _, info = env.step([.5, 0.0])
        self.assertEqual(next_obs["iq"].shape, (2, 4, 256))
        self.assertNotIn("channels", info)
        self.assertNotIn("privileged_state", info)

    def test_power_and_jamming_trends(self):
        env = HapUavEnv(HapUavConfig(max_steps=1))
        env.reset(seed=12)
        channels = env._instantaneous_channels()
        low = env._rates(np.array([.2, 0.0]), channels)
        high = env._rates(np.array([1.0, 0.0]), channels)
        jam = env._rates(np.array([1.0, 1.0]), channels)
        self.assertGreater(high["rate_h"], low["rate_h"])
        self.assertLess(jam["rate_u"], high["rate_u"])

    def test_oracle_is_bounded(self):
        env = HapUavEnv(HapUavConfig(max_steps=1)); env.reset(seed=3)
        action = genie_action(env, step_db=10)
        self.assertTrue(np.all(action >= 0.0)); self.assertTrue(np.all(action <= 1.0))


class DatasetTests(unittest.TestCase):
    def test_generated_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_pilot_hdf5(root / "pilot.h5", 3, HapUavConfig(max_steps=1), 1)
            generate_trajectory_hdf5(root / "traj.h5", 2, 3, HapUavConfig(max_steps=3), 11)
            generate_expert_hdf5(root / "expert.h5", 1, 3, HapUavConfig(max_steps=3), 21, grid_step_db=10)
            pilot = PilotWindowDataset(root / "pilot.h5"); self.assertEqual(pilot[0].shape, (2, 4, 256))
            traj = TrajectoryDataset(root / "traj.h5"); item = traj[0]
            self.assertEqual(item["iq"].shape, (3, 2, 4, 256)); self.assertEqual(item["side"].shape, (3, 8)); self.assertEqual(item["secrecy_rate"].shape, (3,))
            expert = TrajectoryDataset(root / "expert.h5"); self.assertEqual(expert[0]["action"].shape, (3, 2))
            train_a, val_a = _split_pilot_dataset(pilot, 0.33, 123)
            train_b, val_b = _split_pilot_dataset(pilot, 0.33, 123)
            self.assertEqual(len(train_a) + len(val_a), len(pilot))
            self.assertEqual(train_a.indices, train_b.indices)
            self.assertEqual(val_a.indices, val_b.indices)
            self.assertTrue(set(train_a.indices).isdisjoint(val_a.indices))


class ModelTests(unittest.TestCase):
    def test_model_interfaces_and_gradients(self):
        x = torch.randn(1, 2, 4, 256)
        jepa = JEPAWorldModelEncoder(); result = jepa(x)
        self.assertEqual(tuple(jepa.encode_rf(x).shape), (1, 128)); self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        vae = RFVAE(); vae_loss, _ = vae.loss(x); self.assertTrue(torch.isfinite(vae_loss)); vae_loss.backward()
        state = torch.randn(2, 4, 160); action = torch.rand(2, 4, 2)
        dynamics = ActionConditionedMDNLSTM(); output = dynamics(state, action); self.assertEqual(output.mu.shape, (2,4,160,5)); self.assertTrue(torch.isfinite(dynamics.nll(output, torch.randn_like(state))))
        features = torch.randn(2, 416); controller = PowerController(); bounded, logp = controller.action(features); self.assertEqual(bounded.shape, (2,2)); self.assertTrue(torch.all((bounded >= 0) & (bounded <= 1))); self.assertTrue(torch.isfinite(logp).all())


if __name__ == "__main__": unittest.main()
