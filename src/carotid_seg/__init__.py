"""Lightweight attention segmentation of carotid-artery ultrasound images."""

from .model import CoordAtt, build_model
from .training import TrainConfig, compile_model

__all__ = ["CoordAtt", "build_model", "compile_model", "TrainConfig"]
