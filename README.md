# WirelessJEPA

WirelessJEPA is a research codebase for an action-conditioned, multi-timescale
JEPA world model for secrecy-aware AoI/AoLI optimization in RIS-assisted
HAP-IoT networks.

The repository is currently in the architecture and environment-validation
stage. The research rationale and implementation contract are documented in:

- [Research overview](docs/JEPA_World_Model_Research_Overview.md)
- [Architecture and implementation specification](docs/JEPA_World_Model_Architecture_Implementation.md)

## Repository layout

```text
WirelessJEPA/
├── configs/       Experiment and component configuration
├── data/          Rollout collection, datasets, masks, and normalization
├── docs/          Research and architecture documentation
├── environment/   Wireless environment and physical-system simulation
├── evaluation/    Representation, rollout, control, and ablation evaluation
├── models/        Tokenizer, JEPA, dynamics, and policy models
├── tests/         Unit and integration tests
└── training/      Staged training entry points and utilities
```

Implementation should follow the decision gates and staged training order in
the architecture specification. In particular, temporal correlation, action
causality, observation availability, and train/test episode separation must be
validated before policy training begins.
