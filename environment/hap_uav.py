"""Causal partial-observation HAP--UAV secrecy environment.

The environment is deliberately dependency-light.  It exposes a Gym-like
``reset``/``step`` interface but does not require gymnasium, which keeps it
usable from the existing PyTorch/Lightning training jobs.  The simulator
state (including instantaneous eavesdropper channels) is never returned in a
normal observation; it is available only through the explicitly named
``privileged_state`` method for oracle evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class HapUavConfig:
    """Simulation and observation defaults used by the V1 experiments."""

    carrier_frequency_ghz: float = 28.0
    bandwidth_hz: float = 20e6
    slot_duration_s: float = 1.0
    sample_rate_hz: float = 20e6
    pilot_samples: int = 256
    pilot_power_dbm: float = 10.0
    packet_bits: float = 2e6

    hap_altitude_m: float = 20_000.0
    uav_altitude_m: float = 100.0
    uav_initial_xy_m: Tuple[float, float] = (5_000.0, 0.0)
    source_xy_m: Tuple[float, float] = (0.0, 0.0)
    uav_speed_mps: float = 5.0
    trajectory: str = "static"
    max_steps: int = 100

    num_antennas: int = 4
    antenna_spacing_wavelengths: float = 0.5
    atmospheric_loss_db_per_km: float = 0.06
    shadowing_db: float = 6.0
    noise_figure_db: float = 5.0
    source_gain_dbi: float = 8.0
    hap_gain_dbi: float = 20.0
    uav_gain_dbi: float = 1.0
    residual_si_factor: float = 0.35
    jamming_gain_hap: float = 1.0
    jamming_gain_uav: float = 1.0
    jamming_hap_gain_db: float = -80.0

    source_hap_m: int = 3
    source_uav_m: int = 2
    hap_uav_m: int = 2
    jamming_hap_m: int = 1
    jamming_uav_m: int = 1
    tap_delays: Tuple[int, ...] = (0, 2, 5)
    tap_power_db: Tuple[float, ...] = (0.0, -4.0, -8.0)
    fading_correlation: float = 0.95
    phase_noise_std: float = 0.01
    cfo_std_hz: float = 500.0
    awgn: bool = True

    min_power_dbm: float = 0.0
    max_power_dbm: float = 30.0
    secrecy_rate_min_bps: float = 0.5e6
    uav_decode_rate_bps: Optional[float] = None
    age_cap: int = 200
    reward_weights: Tuple[float, float, float, float, float] = (
        1.0,
        0.2,
        0.5,
        0.05,
        1.0,
    )

    seed: int = 42


def dbm_to_watts(power_dbm: Any) -> np.ndarray:
    """Convert dBm to watts without silently treating values as watts."""

    return np.power(10.0, (np.asarray(power_dbm, dtype=np.float64) - 30.0) / 10.0)


def watts_to_dbm(power_watts: Any) -> np.ndarray:
    watts = np.maximum(np.asarray(power_watts, dtype=np.float64), EPS)
    return 10.0 * np.log10(watts) + 30.0


def _complex_normal(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


class HapUavEnv:
    """Four-antenna pilot-I/Q environment for partial-observation control.

    ``observation['iq']`` is ``(2, 4, 256)`` and ``observation['side']`` is
    ``(8,)``.  The side vector is, in order,
    ``[AoI, AoLI, log-gain(SU), log-gain(HU), vx, vy, previous-Pt,
    previous-Pj]``.  All entries are normalized to approximately ``[0, 1]``.
    """

    observation_shape = (2, 4, 256)
    side_shape = (8,)
    action_shape = (2,)

    def __init__(self, config: Optional[HapUavConfig] = None, **overrides: Any):
        self.config = replace(config or HapUavConfig(), **overrides)
        if self.config.num_antennas != 4:
            raise ValueError("V1 requires exactly four HAP receive antennas")
        if len(self.config.tap_delays) != len(self.config.tap_power_db):
            raise ValueError("tap_delays and tap_power_db must have the same length")
        if not 0.0 <= self.config.fading_correlation < 1.0:
            raise ValueError("fading_correlation must be in [0, 1)")
        self.rng = np.random.default_rng(self.config.seed)
        self._reset_state()

    @property
    def horizon(self) -> int:
        return self.config.max_steps

    def _reset_state(self) -> None:
        self.step_index = 0
        self.done = False
        self.age_h = 1
        self.age_u = 1
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.uav_xy = np.asarray(self.config.uav_initial_xy_m, dtype=np.float64).copy()
        self.uav_velocity = np.zeros(2, dtype=np.float64)
        self._channels: Dict[str, np.ndarray] = {}
        self._angles: Dict[str, np.ndarray] = {}
        self._current_iq = np.zeros(self.observation_shape, dtype=np.float32)
        self._current_pilot_channel = None
        self._cfo_hz = 0.0
        self._pilot = None

    def _positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = np.array([*self.config.source_xy_m, 0.0], dtype=np.float64)
        hap = np.array([0.0, 0.0, self.config.hap_altitude_m], dtype=np.float64)
        uav = np.array([*self.uav_xy, self.config.uav_altitude_m], dtype=np.float64)
        return source, hap, uav

    def _distance_km(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / 1000.0)

    def _path_gain(self, distance_km: float, shadow_db: float = 0.0) -> float:
        if distance_km <= 0.0:
            raise ValueError("link distance must be positive")
        f = self.config.carrier_frequency_ghz
        loss_db = (
            92.45
            + 20.0 * np.log10(f)
            + 20.0 * np.log10(distance_km)
            + self.config.atmospheric_loss_db_per_km * distance_km
            + shadow_db
        )
        return float(10.0 ** (-loss_db / 10.0))

    def _noise_watts(self) -> float:
        noise_dbm = -174.0 + 10.0 * np.log10(self.config.bandwidth_hz) + self.config.noise_figure_db
        return float(dbm_to_watts(noise_dbm))

    def _link_geometry(self, name: str) -> Tuple[float, float]:
        source, hap, uav = self._positions()
        if name == "sh":
            return self._distance_km(source, hap), 0.0
        if name == "su":
            return self._distance_km(source, uav), self.config.shadowing_db
        if name in {"hu", "jh", "ju"}:
            return self._distance_km(hap, uav), self.config.shadowing_db if name == "hu" else 0.0
        raise KeyError(name)

    def _make_steering(self, angles: np.ndarray) -> np.ndarray:
        wavelength = 299_792_458.0 / (self.config.carrier_frequency_ghz * 1e9)
        d = self.config.antenna_spacing_wavelengths * wavelength
        antenna_index = np.arange(self.config.num_antennas, dtype=np.float64)
        return np.exp(
            -1j
            * 2.0
            * np.pi
            * antenna_index[None, :]
            * d
            / wavelength
            * np.sin(angles[:, None])
        ) / np.sqrt(self.config.num_antennas)

    def _initialize_channels(self) -> None:
        taps = len(self.config.tap_delays)
        rng = self.rng
        self._angles = {
            "sh": rng.uniform(-np.pi / 3.0, np.pi / 3.0, size=taps),
            "su": rng.uniform(-np.pi, np.pi, size=taps),
            "hu": rng.uniform(-np.pi, np.pi, size=taps),
        }
        # Integer Nakagami m is obtained by averaging m correlated complex
        # Gaussian powers.  This retains a Nakagami-compatible envelope while
        # giving the world model genuine temporal correlation.
        m_values = {
            "sh": self.config.source_hap_m,
            "su": self.config.source_uav_m,
            "hu": self.config.hap_uav_m,
            "jh": self.config.jamming_hap_m,
            "ju": self.config.jamming_uav_m,
        }
        self._channels = {
            name: _complex_normal(rng, (m, taps))
            for name, m in m_values.items()
        }
        self._cfo_hz = float(rng.normal(0.0, self.config.cfo_std_hz))

    def _evolve_channel_state(self) -> None:
        rho = self.config.fading_correlation
        innovation_scale = np.sqrt(max(1.0 - rho * rho, 0.0))
        for name, state in self._channels.items():
            state[...] = rho * state + innovation_scale * _complex_normal(self.rng, state.shape)
        # The source-HAP angle is static in V1.  UAV-related geometry changes
        # through path loss; a small phase walk models unresolved motion.
        for name in ("su", "hu"):
            self._angles[name] = self._angles[name] + self.rng.normal(
                0.0, self.config.phase_noise_std, size=self._angles[name].shape
            )

    def _tap_coefficients(self, name: str, m: int, vector: bool = False) -> np.ndarray:
        if name == "jh":
            # Residual self-interference is an isolated local coupling path.
            gain = 10.0 ** (self.config.jamming_hap_gain_db / 10.0)
        else:
            distance_km, shadow = self._link_geometry(name)
            gain = self._path_gain(distance_km, shadow)
            antenna_gains = {
                "sh": self.config.source_gain_dbi + self.config.hap_gain_dbi,
                "su": self.config.source_gain_dbi + self.config.uav_gain_dbi,
                "hu": self.config.hap_gain_dbi + self.config.uav_gain_dbi,
                "ju": self.config.hap_gain_dbi + self.config.uav_gain_dbi,
            }
            gain *= 10.0 ** (antenna_gains.get(name, 0.0) / 10.0)
        path_power = np.power(10.0, np.asarray(self.config.tap_power_db) / 10.0)
        raw = self._channels[name]
        envelope = np.sqrt(np.mean(np.abs(raw) ** 2, axis=0) + EPS)
        coeff = np.sqrt(gain * path_power) * envelope
        phase = np.exp(1j * np.angle(np.mean(raw, axis=0) + (EPS + 0j)))
        coeff = coeff * phase
        if vector:
            steering = self._make_steering(self._angles["sh"])
            return coeff[:, None] * steering
        return coeff

    def _instantaneous_channels(self) -> Dict[str, np.ndarray]:
        h_sh = self._tap_coefficients("sh", self.config.source_hap_m, vector=True)
        h_su = self._tap_coefficients("su", self.config.source_uav_m)
        h_hu = self._tap_coefficients("hu", self.config.hap_uav_m)
        h_jh = self._tap_coefficients("jh", self.config.jamming_hap_m)
        h_ju = self._tap_coefficients("ju", self.config.jamming_uav_m)
        return {"sh": h_sh, "su": h_su, "hu": h_hu, "jh": h_jh, "ju": h_ju}

    def _generate_pilot(self, channels: Dict[str, np.ndarray]) -> np.ndarray:
        w = self.config.pilot_samples
        if self._pilot is None:
            symbols = self.rng.integers(0, 4, size=w)
            self._pilot = np.exp(1j * (np.pi / 2.0) * symbols).astype(np.complex128)
        pilot = self._pilot
        max_delay = max(self.config.tap_delays)
        impulse = np.zeros((max_delay + 1, self.config.num_antennas), dtype=np.complex128)
        for tap, delay in enumerate(self.config.tap_delays):
            impulse[delay] = channels["sh"][tap]
        received = np.zeros((self.config.num_antennas, w), dtype=np.complex128)
        pilot_power = float(dbm_to_watts(self.config.pilot_power_dbm))
        for antenna in range(self.config.num_antennas):
            received[antenna] = np.convolve(pilot, impulse[:, antenna], mode="full")[:w]
        n = np.arange(w, dtype=np.float64)
        received *= np.sqrt(pilot_power) * np.exp(1j * (2.0 * np.pi * self._cfo_hz * n / self.config.sample_rate_hz))
        if self.config.awgn:
            noise_std = np.sqrt(self._noise_watts() / 2.0)
            received += noise_std * _complex_normal(self.rng, received.shape)
        received /= np.max(np.abs(received)) + EPS
        return np.stack([received.real, received.imag], axis=0).astype(np.float32)

    def _make_side(self, channels: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        channels = channels or self._instantaneous_channels()
        beta_su = self._path_gain(*self._link_geometry("su"))
        beta_hu = self._path_gain(*self._link_geometry("hu"))
        gain_scale = lambda value: float(np.clip((10.0 * np.log10(value + EPS) + 180.0) / 180.0, 0.0, 1.0))
        velocity_scale = max(self.config.uav_speed_mps, 1.0)
        age_scale = np.log1p(self.config.age_cap)
        side = np.array(
            [
                np.log1p(self.age_h) / age_scale,
                np.log1p(self.age_u) / age_scale,
                gain_scale(beta_su),
                gain_scale(beta_hu),
                np.clip(self.uav_velocity[0] / velocity_scale, -1.0, 1.0),
                np.clip(self.uav_velocity[1] / velocity_scale, -1.0, 1.0),
                self.previous_action[0],
                self.previous_action[1],
            ],
            dtype=np.float32,
        )
        return side

    def _observation(self, channels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        return {"iq": self._current_iq.copy(), "side": self._make_side(channels).copy()}

    def reset(
        self,
        seed: Optional[int] = None,
        scenario: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset and return the first *pre-action* pilot observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        if scenario:
            for key, value in scenario.items():
                if not hasattr(self.config, key):
                    raise KeyError(f"Unknown scenario/config field: {key}")
                setattr(self.config, key, value)
            self.uav_xy = np.asarray(self.config.uav_initial_xy_m, dtype=np.float64).copy()
        self._initialize_channels()
        channels = self._instantaneous_channels()
        self._current_pilot_channel = channels["sh"].copy()
        self._current_iq = self._generate_pilot(channels)
        return self._observation(channels), {"seed": seed, "scenario": asdict(self.config)}

    def _update_mobility(self) -> None:
        trajectory = str(self.config.trajectory).lower()
        speed = float(self.config.uav_speed_mps)
        if trajectory == "static":
            self.uav_velocity[:] = 0.0
        elif trajectory == "linear":
            self.uav_velocity[:] = (speed, 0.0)
        elif trajectory == "random_walk":
            angle = self.rng.uniform(-np.pi, np.pi)
            self.uav_velocity[:] = speed * np.array([np.cos(angle), np.sin(angle)])
        elif trajectory == "circular":
            angle = self.step_index * 0.2
            self.uav_velocity[:] = speed * np.array([-np.sin(angle), np.cos(angle)])
        else:
            raise ValueError(f"Unknown UAV trajectory {self.config.trajectory!r}")
        self.uav_xy += self.config.slot_duration_s * self.uav_velocity

    def _rates(self, action_norm: np.ndarray, channels: Dict[str, np.ndarray]) -> Dict[str, float]:
        action_dbm = self.config.min_power_dbm + action_norm * (
            self.config.max_power_dbm - self.config.min_power_dbm
        )
        pt, pj = dbm_to_watts(action_dbm)
        h_sh = np.sum(channels["sh"], axis=0)
        h_su = np.sum(channels["su"])
        h_hu = np.sum(channels["hu"])
        h_jh = np.sum(channels["jh"])
        h_ju = np.sum(channels["ju"])
        noise = self._noise_watts()
        gamma_h = pt * float(np.vdot(h_sh, h_sh).real) / (noise + self.config.residual_si_factor * pj * abs(h_jh) ** 2 + EPS)
        gamma_u = pt * abs(h_su) ** 2 / (noise + pj * abs(h_ju) ** 2 + EPS)
        rh = self.config.bandwidth_hz * np.log2(1.0 + gamma_h)
        ru = self.config.bandwidth_hz * np.log2(1.0 + gamma_u)
        rs = max(float(rh - ru), 0.0)
        required = self.config.packet_bits / max(self.config.slot_duration_s, EPS)
        decode = self.config.uav_decode_rate_bps or required
        gate = rs >= self.config.secrecy_rate_min_bps
        mu_h = bool(gate and rh >= required)
        mu_u = bool(gate and ru >= decode)
        return {
            "pt_dbm": float(action_dbm[0]),
            "pj_dbm": float(action_dbm[1]),
            "pt_w": float(pt),
            "pj_w": float(pj),
            "sinr_h": float(gamma_h),
            "sinr_u": float(gamma_u),
            "rate_h": float(rh),
            "rate_u": float(ru),
            "secrecy_rate": rs,
            "secrecy_gate": bool(gate),
            "mu_h": mu_h,
            "mu_u": mu_u,
            "required_rate": float(required),
        }

    def preview_action(self, action_norm: Any, channels: Optional[Dict[str, np.ndarray]] = None) -> Tuple[float, Dict[str, Any]]:
        """Score an action without advancing state (used by the genie oracle)."""
        action = np.asarray(action_norm, dtype=np.float64).reshape(-1)
        if action.size != 2:
            raise ValueError(f"action must have shape (2,), got {action.shape}")
        clipped = np.clip(action, 0.0, 1.0)
        metrics = self._rates(clipped, channels or self._instantaneous_channels())
        next_age_h = 1 if metrics["mu_h"] else min(self.age_h + 1, self.config.age_cap)
        next_age_u = 1 if metrics["mu_u"] else min(self.age_u + 1, self.config.age_cap)
        w_h, w_u, w_s, w_p, w_v = self.config.reward_weights
        secrecy_norm = min(metrics["secrecy_rate"] / max(self.config.secrecy_rate_min_bps, EPS), 1.0)
        reward = (
            -w_h * next_age_h / float(self.config.age_cap)
            + w_u * next_age_u / float(self.config.age_cap)
            + w_s * secrecy_norm
            - w_p * float(np.sum(clipped))
            - w_v * float(not metrics["secrecy_gate"])
        )
        return float(reward), {**metrics, "age_h": next_age_h, "age_u": next_age_u, "action_norm": clipped.astype(np.float32)}

    def step(self, action_norm: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one data-phase action after the already observed pilot."""

        if self.done:
            raise RuntimeError("step() called after episode termination; call reset()")
        action = np.asarray(action_norm, dtype=np.float64).reshape(-1)
        if action.size != 2:
            raise ValueError(f"action must have shape (2,), got {action.shape}")
        clipped = np.clip(action, 0.0, 1.0)
        channels = self._instantaneous_channels()
        reward, scored = self.preview_action(clipped, channels)
        metrics = {key: value for key, value in scored.items() if key not in {"age_h", "age_u", "action_norm"}}
        self.age_h = int(scored["age_h"])
        self.age_u = int(scored["age_u"])
        self.previous_action = clipped.astype(np.float32)
        self.step_index += 1
        self._update_mobility()
        self._evolve_channel_state()
        next_channels = self._instantaneous_channels()
        self._current_pilot_channel = next_channels["sh"].copy()
        self._current_iq = self._generate_pilot(next_channels)
        terminated = self.step_index >= self.config.max_steps
        self.done = bool(terminated)
        info = dict(metrics)
        info.update(
            {
                "reward": float(reward),
                "age_h": int(self.age_h),
                "age_u": int(self.age_u),
                "action_norm": clipped.astype(np.float32),
                "action_clipped": bool(np.any(clipped != action)),
                "step": self.step_index,
            }
        )
        # Never include exact channels in the normal transition info.
        return self._observation(next_channels), float(reward), terminated, False, info

    def privileged_state(self) -> Dict[str, Any]:
        """Return simulator-only state for the explicitly labelled genie oracle."""

        channels = self._instantaneous_channels()
        return {
            "step": self.step_index,
            "uav_xy": self.uav_xy.copy(),
            "uav_velocity": self.uav_velocity.copy(),
            "age_h": self.age_h,
            "age_u": self.age_u,
            "channels": {key: value.copy() for key, value in channels.items()},
        }

    def render(self) -> None:
        return None


__all__ = ["HapUavConfig", "HapUavEnv", "dbm_to_watts", "watts_to_dbm"]
