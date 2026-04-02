"""Entities module for household object representations."""

from .object_entity import ObjectEntity, PhysicalProperties, StateProperties, AffordanceProperties
from .enums import ObjectCategory, Material, ContainmentState

__all__ = [
    "ObjectEntity",
    "PhysicalProperties",
    "StateProperties",
    "AffordanceProperties",
    "ObjectCategory",
    "Material",
    "ContainmentState",
]
