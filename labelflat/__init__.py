"""Flatten a wine label from a single photo of a cylindrical bottle."""
from .scene import Scene, make_label
from .rectify import detect, Model
from .pipeline import flatten, score

__all__ = ["Scene", "make_label", "detect", "Model", "flatten", "score"]
