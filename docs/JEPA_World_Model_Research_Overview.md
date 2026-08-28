# JEPA-Based World Models for Wireless Resource Optimization

## High-level research problem overview

## 1. Proposed research direction

This research investigates whether a Joint-Embedding Predictive Architecture (JEPA) can provide a more control-relevant latent representation than a reconstruction-trained variational autoencoder (VAE) inside a wireless world model.

The initial application is the secrecy-aware RIS-assisted HAP-IoT system developed by Sadiq *et al.* [R1]. In that system, a ground IoT source communicates with a HAP through an RIS while a UAV attempts to intercept the update packets. A hierarchical controller jointly manages:

- slow-timescale RIS phase configurations;
- slot-level source transmit power;
- slot-level HAP jamming power;
- information freshness at the HAP, measured through AoI;
- information staleness at the eavesdropper, measured through AoLI;
- secrecy-capacity constraints.

The current solution uses hierarchical model-free deep reinforcement learning. The proposed work adds an explicit learned model of the environment in latent space and uses that model to support prediction, imagination, and resource optimization.

---

## 2. Central hypothesis

The hypothesis is:

> A JEPA encoder trained to predict wireless structure in representation space may learn a more control-relevant state than a VAE trained to reconstruct all observation details. When combined with action-conditioned latent dynamics, this state may improve sample efficiency, robustness, or generalization for secrecy-aware AoI/AoLI resource allocation.

This is deliberately phrased as a hypothesis. JEPA is not assumed to be inherently superior to a VAE. The two encoders must be compared using the same information, latent size, dynamics model, controller, reward, action space, and data budget.

---

## 3. Why the problem is worth studying

### 3.1 Limitation of the current model-free controller

The existing HDRL controller learns a direct mapping from observed network states to actions [R1]. It does not explicitly learn:

- how the wireless environment evolves;
- how alternative actions change future AoI/AoLI;
- how UAV mobility affects future secrecy conditions;
- how to simulate candidate future resource allocations without executing them in the real environment.

Consequently, every new policy improvement depends primarily on additional environment interaction.

### 3.2 Limitation of reconstruction-based world models

The classical World Models architecture uses a VAE for observation compression, an MDN-RNN for temporal dynamics, and a controller for action selection [R2]. PlaNet and Dreamer later developed stronger stochastic latent-dynamics and imagination-based control methods [R8-R10].

A VAE is trained to retain enough information to reconstruct the input. In wireless resource allocation, some accurately reconstructable variations may be irrelevant to the control task. Reconstruction can therefore spend representational capacity on low-level or nuisance information instead of predictable quantities that determine future reward and constraints.

MuDreamer provides related evidence that reconstruction-free predictive world models can improve robustness when observations contain distracting information [R11]. This does not prove that JEPA will be better in the proposed wireless system, but it supports testing a prediction-focused alternative.

### 3.3 Why JEPA is relevant

JEPA learns by predicting target representations from visible context rather than reconstructing the original input [R3]. V-JEPA extends feature prediction to temporal observations [R4]. WirelessJEPA and LatentWave show that JEPA objectives can be adapted to multi-antenna IQ, spectrogram, and CSI data using physically meaningful masks and wireless tokenization [R5, R6].

The proposed work differs from those wireless representation-learning studies because the learned state is not used only for classification, positioning, beam prediction, or linear probing. It becomes part of an action-conditioned world model for sequential decision-making.

The closely related “From Pixels to CSI” study uses coupled JEPAs to model control and wireless dynamics and then supports wireless resource management through latent imagination [R7]. This paper is particularly important for positioning the proposed work: it establishes that JEPA-based latent dynamics can support resource decisions, while the proposed research focuses on a different environment, different observation construction, hierarchical mixed actions, physical-layer secrecy, and AoI/AoLI objectives.

---

## 4. System model

The initial testbed is an uplink IoT system containing:

- source node \(S\);
- RIS with \(N\) reflecting elements;
- HAP legitimate receiver \(H\);
- mobile UAV eavesdropper \(U\);
- information-bearing transmit power \(P_t\);
- jamming power \(P_j\);
- RIS phase configuration \(\boldsymbol\phi\).

The optimization objective follows [R1]:

\[
\min_{\{P_t(t),P_j(t),\boldsymbol\phi(t)\}}
\overline{\Delta_H}-\omega\overline{\Delta_U},
\]

subject to:

\[
P_{\min}\le P_t(t)+P_j(t)\le P_{\max},
\]

\[
P_t(t),P_j(t)\ge0,
\]

\[
C_s(t)\ge R_s,
\]

and discrete RIS phase constraints.

The existing reward has the general form

\[
r_t=
-\Xi\Delta_H(t)
+\beta\Delta_U(t)
-\lambda_tP_t(t)
-\lambda_jP_j(t),
\]

with an additional penalty for secrecy violation.

---

## 5. Proposed architecture

The complete system has three main components.

### 5.1 Predictive representation model

A JEPA encoder converts a recent observation history into a latent state:

\[
z_t=f_\theta(\tau_t),
\]

where \(\tau_t\) contains recent channel, mobility, AoI/AoLI, and applied RIS information.

Unlike WirelessJEPA or LatentWave [R5, R6], the initial input need not be resized to an image or restricted to IQ/CSI alone. The recommended input is a native sequential tensor of physically grouped environment observations. Masks can hide:

- temporal blocks;
- HAP or UAV link groups;
- RIS-element groups;
- age-related variables;
- recent future blocks.

The context encoder predicts the teacher encoder’s representations of the masked information, following the broad JEPA principle [R3].

### 5.2 Action-conditioned latent dynamics

The dynamics model learns

\[
p(z_{t+1}\mid z_t,h_t,a_t),
\]

not merely \(p(z_{t+1}\mid z_t)\). Conditioning on the action is essential because the objective is to understand how RIS and power decisions affect future network conditions.

A recurrent state can be updated through

\[
h_{t+1}=g_\psi(h_t,z_t,a_t).
\]

The world model predicts:

- the next latent distribution;
- next AoI;
- next AoLI;
- next secrecy capacity or secrecy violation.

The predicted reward is computed from these quantities and the known action. A stochastic latent model is preferred because fading, mobility, and packet outcomes are uncertain. This follows the probabilistic memory model in World Models and the stochastic latent state approach of PlaNet/Dreamer [R2, R8-R10].

### 5.3 Hierarchical controller

The controller retains the two physical timescales of [R1]:

- the high-level policy selects RIS phase groups every \(K\) slots;
- the low-level policy selects \(P_t\) and \(P_j\) every slot.

The low-level policy uses the current JEPA state, recurrent memory, and applied high-level action. The high-level policy uses a pooled slow-timescale recurrent state.

The learned world model can generate short imagined trajectories. The actor and critic can then learn from predicted future latents and KPI-derived rewards, following the general latent-imagination principle of Dreamer [R9, R10].

---

## 6. Why the observation must be carefully designed

Adding more variables does not automatically improve the latent state. The observation should contain sufficient primitive information without including multiple deterministic copies of the same quantity.

Recommended primitive information includes:

- grouped complex cascaded HAP and UAV channels;
- direct source-HAP and source-UAV channels;
- jamming-link gains;
- current AoI and AoLI;
- current RIS group phases;
- the high-level update clock;
- UAV position/velocity only when available to the compared policies.

Potentially redundant combinations include:

- channel coefficients together with derived SINR, rate, and secrecy capacity;
- individual RIS links together with their grouped products;
- exact CSI together with raw IQ from which that CSI is computed;
- repeated or upsampled versions of the same data;
- fixed scenario constants repeated at every slot.

The first comparison should use the same information as the existing HDRL. A later partial-observation extension can restrict the policy to deployable HAP-side measurements while allowing a privileged teacher to access complete simulator state during training.

---

## 7. Temporal modeling requirement

A world model requires meaningful temporal structure. If the simulator independently resamples Nakagami fading at every slot, the next exact channel realization is not predictable from the history.

The environment should therefore contain:

- temporally correlated fading or shadowing;
- continuous UAV trajectories;
- slowly varying geometry/path loss;
- causal AoI/AoLI evolution;
- explicit effects of RIS and power decisions.

The model should predict distributions and control consequences rather than claim deterministic prediction of irreducible random fading.

This is a critical feasibility condition. Without temporal predictability, an RNN may learn only average channel statistics and provide little advantage over the existing controller.

---

## 8. Main research questions

### RQ1: Representation quality

Does JEPA pretraining produce a more control-relevant latent state than VAE reconstruction when both use the same observation history and latent dimension?

### RQ2: Action-conditioned prediction

Does conditioning the latent dynamics on RIS and power actions improve multi-step prediction of AoI, AoLI, secrecy capacity, and return?

### RQ3: Resource optimization

Can a JEPA-based world model improve the AoI/AoLI/secrecy/power tradeoff relative to the original model-free HDRL and a VAE-based world model?

### RQ4: Sample efficiency

Can imagined latent trajectories reduce the number of real environment interactions required to learn an effective policy?

### RQ5: Generalization

Does the predictive latent state improve robustness to unseen UAV trajectories, fading conditions, link distances, or missing/noisy observations?

### RQ6: Multitimescale modeling

Does a two-timescale latent world model align better with hierarchical RIS/power control than a single shared dynamics model?

---

## 9. Intended scientific contribution

The contribution should not be presented as a mechanical replacement of VAE by JEPA. A stronger contribution is:

> An action-conditioned, multi-timescale JEPA world model for mixed discrete-continuous resource allocation under freshness and secrecy objectives.

Potential contribution components are:

1. **Wireless rollout tokenization:** a nonredundant sequential representation of physical channel groups, mobility, age, and control state without image upsampling.
2. **Action-conditioned JEPA dynamics:** future teacher latents are predicted under explicit RIS and power actions.
3. **Control-relevant predictive heads:** AoI, AoLI, and secrecy predictions provide interpretable signals without reconstructing the complete observation.
4. **Multitimescale world model:** fast power-control dynamics and slow RIS-control dynamics are aligned with the existing HDRL hierarchy.
5. **Latent imagination for secure freshness control:** the controller learns from short predicted trajectories rather than only real simulator interaction.
6. **Fair VAE-versus-JEPA study:** identical information, latent size, dynamics, controller, and training budgets isolate the representation objective.

---

## 10. Experimental methodology

### 10.1 Data collection

Collect complete trajectories from:

- random control;
- analytical AO;
- PPO/SAC/TD3;
- H-PPO/H-SAC/H-TD3;
- perturbed trained policies.

Using only trajectories from one optimized policy creates poor action coverage and makes the world model unreliable outside that policy’s state-action distribution.

### 10.2 Training phases

1. Validate temporal correlation and environment transitions.
2. Generate and split trajectory data by episode/scenario.
3. Pretrain the JEPA context/predictor/EMA-teacher system.
4. Freeze the encoder and train action-conditioned stochastic dynamics.
5. Train the hierarchical controller on real transitions.
6. Add short imagined rollouts.
7. Optionally fine-tune upper JEPA layers with a small learning rate.

### 10.3 Required baselines

- original model-free HDRL [R1];
- VAE + identical dynamics + identical controller;
- JEPA encoder + controller without dynamics;
- JEPA world model with real-only policy training;
- JEPA world model with imagined policy training;
- analytical AO and random policies.

### 10.4 Main evaluation metrics

Control quality:

- average AoI;
- average AoLI;
- secrecy-outage probability;
- transmit and jamming power;
- cumulative return;
- constraint violations.

World-model accuracy:

- one-step and multi-step latent error;
- AoI/AoLI prediction error;
- secrecy-capacity error;
- reward error;
- rollout drift and uncertainty.

Learning and deployment:

- real interactions to reach a target return;
- training stability;
- inference latency;
- model size;
- performance on unseen channel and mobility scenarios.

---

## 11. What would count as success

The JEPA world model does not need to dominate every baseline on every metric. A meaningful result would be one or more of:

- a better AoI/AoLI tradeoff at the same power;
- fewer secrecy violations;
- comparable final return with substantially fewer real interactions;
- stronger performance under unseen UAV trajectories or fading parameters;
- better robustness under noisy or missing measurements;
- faster online decision-making than optimization-based methods;
- comparable control with a more transferable frozen representation.

The likely advantage of the world-model approach is sample efficiency and generalization, not guaranteed asymptotic superiority in a fully observed fixed simulator.

---

## 12. Main risks

### 12.1 JEPA may be unnecessary for a simple state vector

If the current state is low-dimensional, fully observed, and nearly Markov, an MLP policy may already be sufficient. This should be tested honestly with frozen/random/MLP representation baselines.

### 12.2 No predictable dynamics

Independent fading destroys next-state predictability. Temporally correlated processes and probabilistic prediction are required.

### 12.3 Privileged information

Exact UAV CSI and AoLI may be unavailable in practice. The first experiment should match the existing baseline; later experiments should distinguish simulator state from deployable observation.

### 12.4 Model exploitation

Policies trained in imagination may exploit inaccuracies. Use short horizons, uncertainty estimates, model ensembles, and regular real-data anchoring.

### 12.5 Representation collapse

JEPA training can fail if the teacher/context/predictor balance or masks are inappropriate. Monitor latent variance, teacher updates, and downstream predictive accuracy.

### 12.6 Dataset coverage

The world model cannot generalize causally to actions absent from the data. Use diverse exploratory policies and evaluate uncertainty outside the training distribution.

### 12.7 Fairness of comparison

The VAE and JEPA systems must share latent size, dynamics capacity, controller, observations, rewards, and interaction budgets.

### 12.8 Added complexity may not translate into control gains

World-model prediction accuracy does not automatically improve resource allocation. Control performance, sample efficiency, and robustness must remain the primary evidence.

---

## 13. Recommended scope for the first implementation

The first prototype should remain deliberately constrained:

1. Retain the original single-source/RIS/HAP/UAV system.
2. Retain the original mixed RIS and power actions.
3. Retain the AoI/AoLI/secrecy/power reward.
4. Use the same observation information as the current HDRL.
5. Replace image-style input with a native observation-history tensor.
6. Pretrain one temporal JEPA encoder.
7. Train one action-conditioned probabilistic GRU.
8. Predict next latent, AoI, AoLI, and secrecy capacity.
9. Warm-start the existing hierarchical actor-critic using real data.
10. Add short imagined trajectories only after the real-only version is stable.

This scope is sufficient to answer whether predictive representation learning and latent dynamics provide value before extending to partial observation, multiple RISs, multiple users, or hardware data.

---

## 14. Preliminary research statement

> This work develops a JEPA-based wireless world model for secrecy-aware information freshness optimization in RIS-assisted HAP-IoT networks. A temporal JEPA encoder learns latent states from structured observation rollouts without reconstructing the raw input. An action-conditioned recurrent model predicts future latent states and control-relevant quantities under RIS and power decisions. A hierarchical actor-critic then learns slow-timescale RIS configurations and fast-timescale transmit/jamming powers using both real and imagined trajectories. The method is evaluated against the original model-free HDRL and a matched VAE-based world model in terms of AoI, AoLI, secrecy, power, sample efficiency, and generalization.

---

## References

- **[R1]** M. Sadiq, M. S. Haider, A. Fatima, M. S. J. Solaija, H. Jung, and S. A. Hassan, “Intent-Driven Hierarchical DRL for Secrecy-Aware AoI-AoLI Optimization in RIS-Assisted HAP-IoT Communications,” *IEEE Internet of Things Journal*, vol. 13, no. 9, 2026. [DOI: 10.1109/JIOT.2025.3648959](https://doi.org/10.1109/JIOT.2025.3648959).
- **[R2]** D. Ha and J. Schmidhuber, “World Models,” 2018. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122).
- **[R3]** M. Assran *et al.*, “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” ICCV 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
- **[R4]** A. Bardes *et al.*, “Revisiting Feature Prediction for Learning Visual Representations from Video,” 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).
- **[R5]** V. Chu, O. Mashaal, and H. Abou-Zeid, “WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions,” 2026. [arXiv:2601.20190](https://arxiv.org/abs/2601.20190).
- **[R6]** A. Mohamed, A. Aboulfotouh, and H. Abou-Zeid, “LatentWave: JEPA Pretraining for Wireless Foundation Models,” 2026. [arXiv:2606.06373](https://arxiv.org/abs/2606.06373).
- **[R7]** C. Bou Chaaya, A. M. Girgis, and M. Bennis, “From Pixels to CSI: Distilling Latent Dynamics For Efficient Wireless Resource Management,” 2025. [arXiv:2506.16216](https://arxiv.org/abs/2506.16216).
- **[R8]** D. Hafner *et al.*, “Learning Latent Dynamics for Planning from Pixels,” 2019. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551).
- **[R9]** D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” 2020. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603).
- **[R10]** D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering Diverse Domains through World Models,” 2023/2024. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104).
- **[R11]** M. Burchi and R. Timofte, “MuDreamer: Learning Predictive World Models without Reconstruction,” 2024. [arXiv:2405.15083](https://arxiv.org/abs/2405.15083).
- **[R12]** H. Chai, Y. Yuan, and Y. Li, “MobiWorld: World Models for Mobile Wireless Network,” 2025. [arXiv:2507.09462](https://arxiv.org/abs/2507.09462).

