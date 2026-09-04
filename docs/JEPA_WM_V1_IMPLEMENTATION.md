# JEPA+WM V1 implementation

This branch implements the first controlled experiment from the attached design: a partial-observation, non-RIS HAP–UAV environment with four-antenna pilot I/Q, a WirelessJEPA representation, action-conditioned MDN-LSTM dynamics, and real-environment PPO power control. Alternative representation baselines are optional and are not part of this JEPA run.

The full RIS semantic-token/hierarchical controller described in the older tracked design remains a future extension. V1 deliberately keeps only the two continuous actions `(P_t, P_j)`.

## Quick smoke test

```bash
PYTHONPATH=. python data/generate_hap_uav.py --smoke --output-dir /tmp/hap_uav_smoke
PYTHONPATH=. python training/jepa_wm.py jepa --data /tmp/hap_uav_smoke/pilot_train.h5 --output /tmp/hap_uav_smoke/jepa.pt --epochs 1 --batch-size 2 --val-repeats 2 --warmup-epochs 1
PYTHONPATH=. python training/jepa_wm.py mdn --data /tmp/hap_uav_smoke/trajectories_train.h5 --representation jepa --representation-checkpoint /tmp/hap_uav_smoke/jepa.pt --output /tmp/hap_uav_smoke/mdn.pt --epochs 1 --batch-size 1
PYTHONPATH=. python training/jepa_wm.py ppo --representation jepa --representation-checkpoint /tmp/hap_uav_smoke/jepa.pt --dynamics-checkpoint /tmp/hap_uav_smoke/mdn.pt --output /tmp/hap_uav_smoke/ppo.pt --episodes 2 --episode-length 4 --memory
```

Run regression tests with:

```bash
python -m unittest tests.test_jepa_wm_v1 -v
```

## Staged experiment

1. Generate independent-seed synthetic pilot windows and ordered trajectories with `data/generate_hap_uav.py`.
2. Train JEPA (`multi-block` by default) on the raw windows. Alternative representation baselines are intentionally deferred. The full-data GPU launcher uses a deterministic 90/10 pilot split, fixed `.5` antenna ratios unless an explicit sweep is requested, four seeded validation-mask passes, a five-epoch learning-rate warm-up, frozen BatchNorm running statistics, and patience-based early stopping:
   `sbatch jepa-wm-gpu.slurm`
3. Train the frozen-JEPA MDN-LSTM and side-information MLP with `training/jepa_wm.py mdn`; the full A100 launcher is `sbatch mdn-wm-gpu.slurm`.
4. Train the frozen JEPA controller with real-environment PPO (`--memory` for the MDN state; omit it for the no-memory ablation). The default controller horizon is 5,000 slots per episode; PPO updates every 512 transitions with recurrent state carried between rollout chunks. The stabilized launcher uses actor/critic learning rates `1e-4/5e-4`, value-target normalization and clipping, first-minibatch KL stopping, entropy annealing to zero, and periodic deterministic held-out evaluation. The full 500-episode launcher writes versioned `ppo_jepa_5000_v2.pt` plus its `_best.pt` checkpoint.
5. Evaluate random, fixed, balanced, and full-state one-step genie policies with `evaluation/hap_uav_eval.py`; add learned-policy evaluation after the PPO checkpoints are available.
6. Plot CSV reports with `analysis/plotting/scripts/plot_jepa_wm_results.py`.

The simulator returns only `iq: [2,4,256]` and `side: [8]` to normal policies. Instantaneous UAV channels exist only in `HapUavEnv.privileged_state()` for the explicitly labelled oracle. Pilot generation precedes every action, and pilot/data use the same current channel realization. HDF5 datasets are raw and antenna upsampling is performed only in the model.

### Convergence diagnostics

JEPA CSV/TensorBoard logs include the averaged validation loss, validation-repeat count, effective learning rate, and mask configuration. Antenna/time ablations should be compared with the same fixed ratio before running the registered `.25/.5/.75` ratio sweep. The controller logs entropy coefficient, normalized value-target scale, deterministic evaluation return/KPIs, `approx_kl`, `clip_fraction`, action log-standard-deviations, and rollout count. A stochastic episode return is not used as the best-checkpoint criterion; `_best.pt` is selected by deterministic held-out return.
