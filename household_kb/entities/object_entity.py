"""ObjectEntity dataclass representing a household object with all its properties."""

from dataclasses import dataclass, field
from typing import Optional, List
from .enums import ObjectCategory, Material, ContainmentState


@dataclass
class PhysicalProperties:
    """Physical dimensions and properties of an object."""
    weight: float = 0.0          # in grams
    width: float = 0.0           # in centimeters
    height: float = 0.0         # in centimeters
    depth: float = 0.0          # in centimeters
    volume_capacity: float = 0.0  # in milliliters, for containers
    material: Material = Material.MISC


@dataclass
class StateProperties:
    """Mutable state properties of an object."""
    is_open: bool = False
    is_on: bool = False          # for appliances/toggles
    is_full: bool = False
    fill_level: float = 0.0      # 0.0 to 1.0
    is_dirty: bool = False
    is_heated: bool = False
    is_cooled: bool = False
    is_clean: bool = True


@dataclass
class AffordanceProperties:
    """Affordance properties describing what can be done with the object."""
    can_contain: bool = False
    can_support: bool = False
    can_cut: bool = False
    can_heat: bool = False
    can_cool: bool = False
    can_pick: bool = False
    can_open: bool = False
    can_close: bool = False
    can_toggle: bool = False
    can_slice: bool = False
    can_fill: bool = False
    can_pour: bool = False


@dataclass
class ObjectEntity:
    """
    Represents a household object entity in the knowledge base.

    Attributes:
        id: Unique identifier for the object
        name: Human-readable name (e.g., "Apple", "Microwave")
        category: High-level category
        tags: Additional descriptive tags for flexible querying
        physical: Physical properties (weight, dimensions, material)
        state: Current state properties
        affordance: What actions can be performed on this object
    """
    id: str
    name: str
    category: ObjectCategory
    tags: List[str] = field(default_factory=list)
    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    state: StateProperties = field(default_factory=StateProperties)
    affordance: AffordanceProperties = field(default_factory=AffordanceProperties)

    def matches_vague_reference(self, reference: str) -> bool:
        """
        Check if a vague reference (e.g., 'fruit', 'container', 'electronic')
        matches this object through its category or tags.
        """
        reference_lower = reference.lower()
        return (
            reference_lower in self.name.lower()
            or reference_lower in self.category.name.lower()
            or any(reference_lower in tag.lower() for tag in self.tags)
        )

    def get_attr(self, attr_path: str):
        """
        Get an attribute value by dot-separated path.

        Args:
            attr_path: Dot-separated path to attribute (e.g., "physical.weight", "state.is_open")

        Returns:
            Attribute value, or None if path is invalid

        Examples:
            entity.get_attr("name")           -> "Apple"
            entity.get_attr("physical.weight") -> 180.0
            entity.get_attr("state.is_open")   -> False
        """
        parts = attr_path.split(".")
        obj = self
        for part in parts:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def __hash__(self):
        return hash(self.id)
