"""Household Embodied-Agent Knowledge Base.

A lightweight knowledge base for vague instruction resolution in household embodied agents.
"""

from entities import (
    ObjectEntity,
    ObjectCategory,
    Material,
    PhysicalProperties,
    StateProperties,
    AffordanceProperties,
)
from knowledge_base import KnowledgeBase, QueryCriteria

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "ObjectEntity",
    "KnowledgeBase",
    "QueryCriteria",
    # Enums
    "ObjectCategory",
    "Material",
    # Property dataclasses
    "PhysicalProperties",
    "StateProperties",
    "AffordanceProperties",
]
