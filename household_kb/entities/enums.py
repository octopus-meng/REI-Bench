"""Enumerations for household object categories and materials."""

from enum import Enum, auto


class ObjectCategory(Enum):
    """High-level categories for household objects."""
    FOOD = auto()
    UTENSIL = auto()
    APPLIANCE = auto()
    FURNITURE = auto()
    CONTAINER = auto()
    ELECTRONIC = auto()
    TEXTILE = auto()
    CLEANING = auto()
    DECOR = auto()
    MISC = auto()


class Material(Enum):
    """Physical materials of objects."""
    METAL = auto()
    PLASTIC = auto()
    GLASS = auto()
    CERAMIC = auto()
    WOOD = auto()
    FABRIC = auto()
    PAPER = auto()
    RUBBER = auto()
    FOOD = auto()
    COMPOSITE = auto()
    MISC = auto()


class ContainmentState(Enum):
    """State of containment for containers."""
    EMPTY = 0.0
    PARTIAL = 0.5
    FULL = 1.0
    UNKNOWN = -1.0
