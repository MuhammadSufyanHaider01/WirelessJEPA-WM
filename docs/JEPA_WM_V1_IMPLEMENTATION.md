# JEPA+WM V1 implementation

This branch implements the first controlled experiment from the attached design: a partial-observation, non-RIS HAP–UAV environment with four-antenna pilot I/Q, a WirelessJEPA representation, an equal-latent VAE baseline, action-conditioned MDN-LSTM dynamics, and real-environment PPO power control.

The full RIS semantic-token/hierarchical controller described in the older tracked design remains a future extension. V1 deliberately keeps only the two continuous actions `(P_t, P_j)`.

## Quick smoke test

```bash
PYTHONPATH=. python data/generate_hap_uav.py --smoke --output-dir /tmp/hap_uav_smoke
PYTHONPATH=. python training/jepa_wm.py jepa --data /tmp/hap_uav_smoke/pilot_train.h5 --output /tmp/hap_uav_smoke/jepa.pt --epochs 1 --batch-size 2
PYTHONPATH=. python training/jepa_wm.py vae --data /tmp/hap_uav_smoke/pilot_train.h5 --output /tmp/hap_uav_smoke/vae.pt --epochs 1 --batch-size 2
PYTHONPATH=. python training/jepa_wm.py mdn --data /tmp/hap_uav_smoke/trajectories_train.h5 --representation jepa --representation-checkpoint /tmp/hap_uav_smoke/jepa.pt --output /tmp/hap_uav_smoke/mdn.pt --epochs 1 --batch-size 1
PYTHONPATH=. python training/jepa_wm.py ppo --representation jepa --representation-checkpoint /tmp/hap_uav_smoke/jepa.pt --dynamics-checkpoint /tmp/hap_uav_smoke/mdn.pt --output /tmp/hap_uav_smoke/ppo.pt --episodes 2 --episode-length 4 --memory
```

Run regression tests with:

```bash
python -m unittest tests.test_jepa_wm_v1 -v
```

## Staged experiment

1. Generate independent-seed synthetic pilot windows and ordered trajectories with `data/generate_hap_uav.py`.
2. Train JEPA (`multi-block` by default) and the matched VAE on the same raw windows. The full-data RTX 6000 launcher uses a deterministic 90/10 pilot split, saves the best validation checkpoint, and enables patience-based early stopping:
   `sbatch jepa-wm-gpu.slurm`
3. Train the frozen-representation MDN-LSTM with `training/jepa_wm.py mdn`.
4. Train the frozen JEPA/VAE controller with real-environment PPO (`--memory` for the MDN state; omit it for the no-memory ablation).
5. Evaluate random, fixed, balanced, and full-state one-step genie policies with `evaluation/hap_uav_eval.py`; add learned-policy evaluation after the PPO checkpoints are available.
6. Plot CSV reports with `analysis/plotting/scripts/plot_jepa_wm_results.py`.

The simulator returns only `iq: [2,4,256]` and `side: [8]` to normal policies. Instantaneous UAV channels exist only in `HapUavEnv.privileged_state()` for the explicitly labelled oracle. Pilot generation precedes every action, and pilot/data use the same current channel realization. HDF5 datasets are raw and antenna upsampling is performed only in the model.
