# JEPA World Model for RIS-Assisted HAP-IoT Control

## Architecture and implementation specification

**Purpose.** This document is a coding-oriented specification for an action-conditioned, multi-timescale JEPA world model applied to secrecy-aware AoI/AoLI resource allocation in the RIS-assisted HAP-IoT system of Sadiq *et al.* [R1]. It fixes the data interface, tensor dimensions, module boundaries, losses, training order, baselines, and implementation checks needed for a first reproducible prototype.

The numerical dimensions are starting values, not final claims. They must later be tested through ablations.

---

## 1. Design objective

The proposed model replaces the reconstruction-trained VAE in the classical World Models pipeline [R2] with a JEPA encoder. The encoder learns a compact state from wireless observation histories without reconstructing every input feature. An action-conditioned recurrent model predicts the next latent distribution and the control-relevant network quantities. A hierarchical controller then selects RIS configurations and transmit/jamming powers.

The intended flow is

\[
\tau_t \xrightarrow{\text{JEPA}} z_t,
\]

\[
(z_t,h_t,a_t)
\xrightarrow{\text{latent dynamics}}
(\hat z_{t+1},\widehat{\Delta_H},\widehat{\Delta_U},\widehat C_s),
\]

\[
(z_t,h_t)\xrightarrow{\text{hierarchical controller}}a_t.
\]

The dynamics model **must be conditioned on the action**. A model of only \(p(z_{t+1}\mid z_t)\) cannot learn the causal consequences of RIS and power decisions.

---

## 2. Source system and optimization interface

The source system [R1] contains:

- one ground IoT source \(S\);
- one RIS with \(N=64\) elements;
- a HAP legitimate receiver \(H\);
- a mobile UAV eavesdropper \(U\);
- a high-level controller that updates RIS phases every \(K\) slots;
- a low-level controller that selects transmit and jamming powers every slot;
- AoI at the HAP, \(\Delta_H\);
- AoLI at the UAV, \(\Delta_U\);
- an instantaneous secrecy-capacity constraint \(C_s\ge R_s\).

The environment should expose transitions of the form

```text
(observation_t, action_t, kpi_t, reward_t, observation_t+1, done_t)
```

where the action is

\[
a_t=(a_k^H,a_t^L),
\qquad
a_k^H=\boldsymbol\phi_k,
\qquad
a_t^L=(P_t,P_j).
\]

The proposed model should initially use the **same observable information, action space, constraints, and reward as the existing HDRL baseline**. Giving JEPA additional privileged variables would invalidate the encoder comparison.

---

## 3. Default configuration

| Symbol | Meaning | Initial value |
|---|---|---:|
| \(B\) | minibatch size | 64 |
| \(N\) | RIS elements | 64 |
| \(G\) | RIS phase groups | 8 |
| \(N/G\) | elements per group | 8 |
| \(Q\) | phase choices per group | 4 |
| \(W\) | observation-history length | 16 slots |
| \(M\) | semantic tokens per slot | 20 |
| \(S=WM\) | physical tokens per history | 320 |
| \(d_z\) | JEPA/token embedding | 128 |
| \(d_h\) | recurrent hidden state | 256 |
| \(d_{\mathrm{ff}}\) | Transformer feedforward width | 512 |
| \(L_c\) | context/teacher layers | 6 |
| \(L_p\) | JEPA predictor layers | 3 |
| \(H_a\) | attention heads | 4 |
| \(\rho\) | initial mask ratio | 0.5 |
| \(H_{\mathrm{imag}}\) | initial imagination horizon | 5-10 slots |

`K` should remain an environment/configuration parameter because it represents the physical high-level RIS update period used in the current system.

---

## 4. Minimal nonredundant per-slot observation

### 4.1 RIS-group cascaded channels

If every element in RIS group \(\mathcal G_g\) shares a phase, define

\[
c_{H,g}(t)=\sum_{n\in\mathcal G_g}g_n(t)u_{H,n}(t),
\]

\[
c_{U,g}(t)=\sum_{n\in\mathcal G_g}g_n(t)u_{U,n}(t).
\]

Store real and imaginary components. For \(G=8\):

| Feature | Dimension |
|---|---:|
| HAP cascaded channels | \(2G=16\) |
| UAV cascaded channels | \(2G=16\) |
| Total | \(4G=32\) |

Do not store \(g_n\), \(u_{H,n}\), \(u_{U,n}\), and their grouped products simultaneously. The separate components are redundant once the grouped products are used for group-level phase control.

### 4.2 Direct and jamming links

Use

\[
[\Re h_{SH},\Im h_{SH},\Re h_{SU},\Im h_{SU},
\log G_{JH},\log G_{JU}],
\]

with dimension 6.

### 4.3 Information-age state

Use

\[
[\log(1+\Delta_H),\log(1+\Delta_U)],
\]

with dimension 2. The log transform limits domination by unusually large ages.

### 4.4 Current RIS state

Represent each applied group phase using sine and cosine:

\[
[\sin\phi_1,\cos\phi_1,\ldots,\sin\phi_G,\cos\phi_G].
\]

Add the normalized high-level clock

\[
\kappa_t=(t\bmod K)/K.
\]

For \(G=8\), this block has dimension \(2G+1=17\).

### 4.5 UAV mobility

If the same variables are available to the existing baseline, include normalized

\[
[x_U,y_U,v_{x,U},v_{y,U}],
\]

with dimension 4. Position and velocity are not equivalent to instantaneous channel gains because they distinguish different future motion directions.

If these variables cannot be estimated by the deployed agent, set `use_mobility_token=false` and remove them from **all** compared policies.

### 4.6 Observation dimension

With mobility:

\[
d_o=4G+6+2+(2G+1)+4=6G+13.
\]

For \(G=8\):

\[
\boxed{o_t\in\mathbb R^{61}}.
\]

Without mobility:

\[
d_o=6G+9=57.
\]

### 4.7 Explicitly excluded inputs

Do not simultaneously include:

- SINR, rates, secrecy capacity, and the primitive channels/actions from which they are derived;
- raw per-element channels and grouped cascaded channels;
- reward values;
- the current action inside `observation_t`;
- duplicated, interpolated, or upsampled copies of the same measurements;
- fixed system constants unless they are randomized across episodes;
- raw IQ and exact CSI describing the same received samples.

If scenario parameters such as \(R_s\), \(P_{\max}\), path-loss parameters, or UAV speed are randomized, add one scenario token containing only the parameters that change.

---

## 5. Semantic tokenization

The history tensor before tokenization is

```text
observation_history: [B, W, d_o] = [B, 16, 61]
```

Create the following semantic tokens at every slot:

| Token type | Count | Raw features per token |
|---|---:|---:|
| HAP RIS-group channel | 8 | 2 |
| UAV RIS-group channel | 8 | 2 |
| Direct/jamming links | 1 | 6 |
| AoI/AoLI | 1 | 2 |
| Applied RIS state and clock | 1 | 17 |
| UAV mobility | 1 | 4 |
| **Total** | **20** | variable |

Thus

\[
M=2G+4=20,
\qquad
S=WM=16\times20=320.
\]

Use a type-specific projector for each token type:

```text
channel_projector: Linear(2, 128)
link_projector:    MLP(6, 128, 128)
age_projector:     MLP(2, 64, 128)
ris_projector:     MLP(17, 128, 128)
mobility_projector: MLP(4, 64, 128)
```

After projection:

```text
tokens: [B, 16, 20, 128]
flattened_tokens: [B, 320, 128]
```

Prepend one learned state token:

```text
jepa_input: [B, 321, 128]
```

Add:

- temporal position embedding `[16, 128]`;
- token-type embedding `[20, 128]`;
- RIS-group identity embedding `[8, 128]`.

No image conversion or spatial resizing is required. This follows the JEPA principle of predicting representations [R3, R4] while adapting token semantics to wireless system variables, as wireless JEPA work adapts masking and tokenization to physical signal structure [R5, R6].

---

## 6. Mask generator

Implement separate mask families rather than one generic random mask:

1. **Temporal block:** mask consecutive slots for selected token types.
2. **Link block:** mask HAP- or UAV-related token groups.
3. **RIS-group block:** mask corresponding HAP/UAV tokens for selected RIS groups.
4. **KPI block:** mask the AoI/AoLI token.
5. **Future block:** hide the most recent slots and predict their teacher embeddings.

The state token is never masked.

At a 50% mask ratio:

```text
physical tokens: 320
visible tokens:  approximately 160
masked tokens:   approximately 160
context input including state token: [B, 161, 128]
```

Record the exact visible and target indices for every sample. The predictor and teacher target gather must use the same target-index tensor.

---

## 7. JEPA modules and dimensions

### 7.1 Context encoder

Initial specification:

```text
layers: 6
embedding dimension: 128
attention heads: 4
head dimension: 32
feedforward dimension: 512
dropout: 0.0-0.1
```

During masked pretraining:

```text
input:  [B, 1 + N_visible, 128]
output: [B, 1 + N_visible, 128]
```

During online control, use the complete history:

```text
input:  [B, 321, 128]
output: [B, 321, 128]
state latent z_t = output[:, 0, :]
z_t: [B, 128]
```

### 7.2 Teacher encoder

The teacher has the same architecture as the context encoder. It receives the complete unmasked token history and is updated by exponential moving average:

\[
\xi\leftarrow m\xi+(1-m)\theta.
\]

Use an EMA momentum schedule starting near 0.996 and approaching 1.0. The teacher receives no gradient.

```text
teacher input:  [B, 321, 128]
teacher output: [B, 321, 128]
selected masked targets: [B, N_mask, 128]
```

### 7.3 JEPA predictor

Initial specification:

```text
layers: 3
embedding dimension: 128
attention heads: 4
feedforward dimension: 512
```

The predictor receives context embeddings, learned mask tokens, and the target positions. It returns only the target positions:

```text
prediction: [B, N_mask, 128]
teacher target: [B, N_mask, 128]
```

Use normalized smooth-L1 or cosine distance:

\[
\mathcal L_{\mathrm{JEPA}}
=
\frac{1}{N_{\mathrm{mask}}}
\sum_i
\left[1-\cos(\hat y_i,\operatorname{sg}(y_i^*))\right].
\]

The basic teacher/context/predictor separation follows I-JEPA [R3], while temporal feature prediction is motivated by V-JEPA [R4] and wireless adaptations in WirelessJEPA and LatentWave [R5, R6].

---

## 8. Action representation

### 8.1 High-level action

The HLC selects one phase index per RIS group:

\[
a_k^H\in\{0,\ldots,Q-1\}^{G}.
\]

With \(G=8,Q=4\):

```text
HLC actor logits: [B, 8, 4]
sampled phase indices: [B, 8]
```

Use a factorized categorical distribution. Do not create one categorical action with \(Q^G\) classes.

Convert selected phases to sine/cosine:

```text
phase_encoding: [B, 16]
phase_action_embedding: MLP(16, 64, 64) -> [B, 64]
```

### 8.2 Low-level power action

The LLC outputs two raw values:

```text
raw_power_action: [B, 2]
```

Enforce the power constraints analytically:

\[
P_{\mathrm{tot}}
=P_{\min}+(P_{\max}-P_{\min})\sigma(u_1),
\]

\[
s=\sigma(u_2),
\qquad
P_t=sP_{\mathrm{tot}},
\qquad
P_j=(1-s)P_{\mathrm{tot}}.
\]

Then

```text
power_action: [B, 2]
power_action_embedding: MLP(2, 32, 32) -> [B, 32]
```

### 8.3 Combined action

Raw dynamics action:

```text
combined_action = [phase_sin_cos, P_t, P_j]
combined_action: [B, 18]
```

Embedded action:

```text
action_embedding = concat([phase_embedding, power_embedding])
action_embedding: [B, 96]
```

---

## 9. Action-conditioned stochastic dynamics

The world model should learn

\[
p(z_{t+1},\Delta_H(t+1),\Delta_U(t+1),C_s(t+1)
\mid z_t,h_t,a_t).
\]

GRU input:

```text
z_t: [B, 128]
action_embedding: [B, 96]
gru_input = concat([z_t, action_embedding]): [B, 224]
```

GRU:

```text
GRU input size: 224
GRU hidden size: 256
h_t: [B, 256]
h_t+1: [B, 256]
```

### 9.1 Next-latent distribution

Use

```text
latent_distribution_head: Linear(256, 256)
mu_t+1: [B, 128]
log_sigma_t+1: [B, 128]
sampled_z_hat_t+1: [B, 128]
```

The target is the state token produced by the EMA teacher on the true next history:

```text
teacher_next_state z*_t+1: [B, 128]
```

Possible loss:

\[
\mathcal L_{\mathrm{dyn}}
=-log p_\psi(z^*_{t+1}\mid z_t,h_t,a_t)
+\lambda_{\cos}
[1-\cos(\mu_{t+1},z^*_{t+1})].
\]

### 9.2 Control-KPI head

Predict only

```text
kpi_head: MLP(256, 128, 3)
kpi_prediction: [B, 3]
```

with components

\[
[\widehat{\Delta_H(t+1)},
\widehat{\Delta_U(t+1)},
\widehat{C_s(t+1)}].
\]

Compute the reward using the known reward equation and known power action:

\[
\hat r_t=
-\Xi\widehat{\Delta_H}
+\beta\widehat{\Delta_U}
-\lambda_tP_t
-\lambda_jP_j
-\lambda_s\mathbf 1[\widehat C_s<R_s].
\]

This avoids simultaneously predicting both the KPIs and a mathematically derived reward.

### 9.3 Total world-model loss

\[
\mathcal L_{\mathrm{WM}}
=\mathcal L_{\mathrm{dyn}}
+\lambda_H\mathcal L_{\Delta_H}
+\lambda_U\mathcal L_{\Delta_U}
+\lambda_s\mathcal L_{C_s}.
\]

Use Huber losses for age values and a scale-appropriate regression loss for secrecy capacity. A binary secrecy-outage auxiliary head can be added only if the continuous \(C_s\) head is poorly calibrated.

Stochastic latent dynamics are motivated by the MDN-RNN in World Models [R2] and the deterministic/stochastic latent state design in PlaNet and Dreamer [R8, R9].

---

## 10. Controller dimensions

### 10.1 Low-level controller

Input:

```text
z_t: [B, 128]
h_t: [B, 256]
current applied phase encoding: [B, 16]
LLC state: [B, 400]
```

Actor:

```text
400 -> 256 -> 128 -> 2
```

Output:

```text
raw power parameters: [B, 2]
constrained action [P_t, P_j]: [B, 2]
```

PPO value network:

```text
400 -> 256 -> 128 -> 1
V_L: [B, 1]
```

### 10.2 High-level controller

Collect recurrent states across one high-level period:

```text
high_level_hidden_sequence: [B, K, 256]
attention/mean pooled hidden: [B, 256]
latest z: [B, 128]
HLC state: [B, 384]
```

Actor:

```text
384 -> 256 -> 128 -> G*Q
```

For \(G=8,Q=4\):

```text
flat logits: [B, 32]
reshaped logits: [B, 8, 4]
```

PPO value network:

```text
384 -> 256 -> 128 -> 1
V_H: [B, 1]
```

The two-timescale controller preserves the physical decomposition of [R1] rather than replacing it with one monolithic actor.

---

## 11. End-to-end shape trace

```text
Per-slot observation                       [B, 61]
16-slot history                           [B, 16, 61]
Semantic raw tokens                       [B, 16, 20, variable]
Projected tokens                          [B, 16, 20, 128]
Flattened physical tokens                 [B, 320, 128]
Add state token                           [B, 321, 128]
Unmasked context-encoder output           [B, 321, 128]
State latent z_t                          [B, 128]

HLC sampled phase indices                 [B, 8]
HLC factorized logits                     [B, 8, Q]
Phase sine/cosine encoding                [B, 16]
Power action                              [B, 2]
Raw combined action                       [B, 18]
Embedded combined action                  [B, 96]

Dynamics input [z_t, action embedding]    [B, 224]
GRU hidden state                          [B, 256]
Next-latent mean                          [B, 128]
Next-latent log standard deviation        [B, 128]
Sampled next latent                       [B, 128]
Predicted control KPIs                    [B, 3]

LLC policy input                          [B, 400]
LLC action                                [B, 2]
HLC policy input                          [B, 384]
HLC logits                                [B, 8, Q]
```

---

## 12. Recommended training stages

### Stage 0: environment validation

Before learning:

1. Confirm that the channel process has temporal correlation. If Nakagami samples are independent at every slot, exact next-channel prediction is impossible.
2. Preserve continuous UAV motion between slots.
3. Validate AoI/AoLI state transitions with deterministic unit tests.
4. Validate that power and RIS actions causally alter the next KPIs.
5. Separate simulator state from agent-observable state.

### Stage 1: dataset generation

Collect complete episodes using a mixture of:

- random policies;
- analytical AO;
- PPO/SAC/TD3;
- H-PPO/H-SAC/H-TD3;
- perturbed versions of trained policies.

Store complete trajectories rather than shuffled independent transitions. Split train/validation/test by episode and scenario, not by adjacent time steps.

Suggested transition record:

```python
Transition = {
    "observation": float32[d_o],
    "ris_phase_index": int64[G],
    "power_action": float32[2],
    "next_observation": float32[d_o],
    "delta_h_next": float32[1],
    "delta_u_next": float32[1],
    "secrecy_capacity_next": float32[1],
    "reward": float32[1],
    "done": bool,
    "scenario_id": int64,
}
```

### Stage 2: JEPA pretraining

1. Sample contiguous 16-slot windows.
2. Convert each window into 320 semantic tokens.
3. Generate physical masks.
4. Encode visible tokens with the context encoder.
5. Encode the complete window with the EMA teacher.
6. Predict masked teacher tokens.
7. Update context encoder, token projectors, and predictor.
8. Update the teacher using EMA.

### Stage 3: latent-dynamics training

Initially freeze the JEPA context encoder.

For every transition window:

1. encode the current window to \(z_t\);
2. encode the shifted next window with the teacher to \(z^*_{t+1}\);
3. embed the executed action;
4. update the GRU hidden state;
5. predict the next-latent distribution and KPIs;
6. optimize \(\mathcal L_{\mathrm{WM}}\).

Train both one-step and multi-step unrolled losses. Start with 1-5 steps before increasing the rollout horizon.

### Stage 4: controller warm start in the real simulator

Preserve the warm-start principle of [R1]:

1. train the LLC while holding the HLC fixed;
2. train both controllers using real transitions;
3. compare with the original H-PPO before using imagined data.

### Stage 5: imagined actor-critic training

From real encoded states:

1. sample policy actions;
2. advance the learned latent dynamics;
3. compute predicted KPI-derived rewards;
4. train actor and value networks on short imagined trajectories;
5. mix real and imagined batches.

If the policy is trained only on real transitions with JEPA features, label the result **JEPA-enhanced HDRL**, not a complete world-model controller. Dreamer-style imagination is the main reference for using latent dynamics to improve policy learning [R9, R10].

### Stage 6: optional joint fine-tuning

Only after the frozen-encoder system is stable:

- unfreeze the upper context-encoder layers;
- use a learning rate 10-100 times smaller than the dynamics/controller rates;
- retain the JEPA objective to prevent task-specific collapse;
- monitor latent variance and transfer performance.

---

## 13. PyTorch-oriented module skeleton

```python
class WirelessTokenizer(nn.Module):
    def forward(self, obs_history):
        # obs_history: [B, W, 61]
        # return: [B, 320, 128]
        ...


class WirelessJEPA(nn.Module):
    def encode_context(self, tokens, visible_indices=None):
        # full online input: [B, 321, 128]
        # masked pretraining input: [B, 1 + N_visible, 128]
        ...

    @torch.no_grad()
    def encode_teacher(self, full_tokens):
        # [B, 321, 128]
        ...

    def predict_targets(self, context_embeddings, target_indices):
        # [B, N_mask, 128]
        ...

    def state(self, obs_history):
        # return z: [B, 128]
        ...


class ActionEncoder(nn.Module):
    def forward(self, phase_indices, power_action):
        # phase_indices: [B, G]
        # power_action: [B, 2]
        # return raw_action [B, 18], embedded_action [B, 96]
        ...


class LatentDynamics(nn.Module):
    def forward(self, z_t, action_embedding, h_t):
        # z_t: [B, 128]
        # action_embedding: [B, 96]
        # h_t: [B, 256]
        # return h_next [B, 256], mu [B, 128], log_sigma [B, 128], kpi [B, 3]
        ...


class LowLevelActorCritic(nn.Module):
    def forward(self, z_t, h_t, applied_phase_encoding):
        # concat state: [B, 400]
        # return power distribution parameters and V_L [B, 1]
        ...


class HighLevelActorCritic(nn.Module):
    def forward(self, z_latest, hidden_sequence):
        # z_latest: [B, 128]
        # hidden_sequence: [B, K, 256]
        # return logits [B, G, Q] and V_H [B, 1]
        ...
```

---

## 14. Suggested repository structure

```text
project/
├── configs/
│   ├── environment.yaml
│   ├── jepa.yaml
│   ├── dynamics.yaml
│   └── controller.yaml
├── environment/
│   ├── hap_ris_env.py
│   ├── channels.py
│   ├── mobility.py
│   ├── aoi_aoli.py
│   └── reward.py
├── data/
│   ├── collect_rollouts.py
│   ├── trajectory_dataset.py
│   ├── normalization.py
│   └── masks.py
├── models/
│   ├── tokenizer.py
│   ├── jepa.py
│   ├── action_encoder.py
│   ├── dynamics.py
│   ├── high_level_policy.py
│   └── low_level_policy.py
├── training/
│   ├── train_jepa.py
│   ├── train_dynamics.py
│   ├── train_controller_real.py
│   ├── train_controller_imagined.py
│   └── ema.py
├── evaluation/
│   ├── evaluate_representation.py
│   ├── evaluate_rollouts.py
│   ├── evaluate_control.py
│   └── ablations.py
└── tests/
    ├── test_dimensions.py
    ├── test_environment.py
    ├── test_power_constraints.py
    ├── test_mask_alignment.py
    └── test_causality.py
```

---

## 15. Required unit tests and invariants

1. `observation.shape[-1] == 6*G + 13` when mobility is enabled.
2. Token count equals `W*(2*G + 4)`.
3. Context and teacher target indices are aligned exactly.
4. Teacher parameters receive no gradients.
5. EMA update changes teacher parameters after every optimizer step.
6. `P_t >= 0`, `P_j >= 0`, and `P_min <= P_t + P_j <= P_max` for all actor outputs.
7. High-level phase actions change only on valid `t % K` slots.
8. The dynamics prediction changes when the action changes while \(z_t,h_t\) are held fixed.
9. No train/validation/test windows overlap the same episode.
10. Shuffling the action sequence must worsen future-KPI prediction; otherwise the model may be ignoring actions.
11. Shuffling time order must worsen future-latent prediction; otherwise the model may not be using temporal information.
12. Latent batch variance must stay above a monitored minimum to detect collapse.

---

## 16. Baselines and ablations

### Required baselines

1. Original H-PPO/H-SAC/H-TD3 from [R1].
2. VAE encoder + identical GRU + identical hierarchical controller.
3. JEPA encoder + controller without learned dynamics.
4. JEPA + action-conditioned dynamics + real-only controller training.
5. JEPA + action-conditioned dynamics + imagined controller training.

### Required ablations

- no action conditioning;
- deterministic versus probabilistic dynamics;
- one-step versus multi-step prediction;
- random versus physical masking;
- \(W\in\{4,8,16,32\}\);
- \(d_z\in\{64,128,256\}\);
- single-timescale versus hierarchical controller;
- frozen versus fine-tuned JEPA;
- mobility token on/off;
- full observation versus partial observation;
- different RIS grouping sizes.

For the VAE/JEPA comparison, hold the latent dimension, dynamics, controller, action space, dataset, reward, training interactions, and scenario splits constant.

---

## 17. Evaluation metrics

### Control metrics

- average AoI at the HAP;
- average AoLI at the UAV;
- secrecy-outage probability;
- average transmit and jamming power;
- scalar return;
- constraint-violation frequency;
- real-environment interactions required to reach a target performance;
- controller inference time.

### World-model metrics

- one-step and multi-step latent prediction loss;
- AoI/AoLI mean absolute error;
- secrecy-capacity error;
- secrecy-outage classification accuracy/calibration;
- reward error computed from predicted KPIs;
- rollout drift versus horizon;
- disagreement among dynamics ensembles.

### Generalization metrics

- unseen UAV trajectories and speeds;
- unseen Nakagami parameters;
- unseen source/RIS/UAV distances;
- missing/noisy observation variables;
- changed RIS group counts or phase resolution.

The primary success claim should preferably combine control quality with sample efficiency or out-of-distribution robustness. Merely obtaining a marginally better training-environment reward may not justify the additional model complexity.

---

## 18. Main implementation risks

| Risk | Coding/experimental mitigation |
|---|---|
| Independent fading contains no predictable temporal signal | add temporally correlated fading or predict distributions/KPIs rather than exact samples |
| JEPA adds little to a perfectly observed 61-D state | evaluate partial/noisy observations and sample efficiency |
| Policy exploits world-model errors | short imagination horizons, uncertainty ensembles, real/imagined data mixing |
| Dataset lacks action coverage | collect rollouts from diverse and perturbed policies |
| Latent ignores control-relevant information | KPI auxiliary heads and action-conditioned multi-step losses |
| Representation collapse | EMA teacher, target normalization, variance monitoring, diverse masks |
| UAV state is not deployably observable | remove it from all policies or use a privileged teacher only during training |
| RIS action is combinatorial | phase grouping and factorized categorical actions |
| Hierarchical training is unstable | low-level warm start followed by joint training |
| VAE/JEPA comparison is unfair | identical interfaces, latent size, controller, data, and compute budget |

---

## 19. Initial implementation decision gates

Do not begin full policy training until these questions are resolved:

1. Is fading temporally correlated in the current simulator?
2. Which UAV variables are genuinely available to the policy?
3. Is the current baseline already using phase grouping, and what is \(G\)?
4. What is the exact high-level update interval \(K\)?
5. Will the first paper claim focus on final control performance, sample efficiency, generalization, or partial observability?
6. Will the first controller use real-only H-PPO as a diagnostic before adding imagination?

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

