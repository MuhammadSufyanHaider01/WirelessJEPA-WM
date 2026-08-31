"""Dataset readers and synthetic HAP--UAV data generation."""
from .hap_uav import PilotWindowDataset, TrajectoryDataset

__all__ = ["PilotWindowDataset", "TrajectoryDataset"]
