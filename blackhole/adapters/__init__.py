"""Model adapter contracts; no concrete model is bundled."""

from .base import ModelAdapter
from .registry import AdapterRegistry

__all__ = ["AdapterRegistry", "ModelAdapter"]

