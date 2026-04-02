"""YAML loader for constructing a KnowledgeBase from YAML files."""

import yaml
from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass, asdict

from entities import (
    ObjectEntity, ObjectCategory, Material,
    PhysicalProperties, StateProperties, AffordanceProperties,
)
from .knowledge_base import KnowledgeBase


@dataclass
class ObjectEntityYAML:
    """YAML-friendly representation of ObjectEntity."""
    id: str
    name: str
    category: str
    tags: List[str] = None
    physical: dict = None
    state: dict = None
    affordance: dict = None


def _parse_material(value: str) -> Material:
    """Parse material string to Material enum."""
    if value is None:
        return Material.MISC
    try:
        return Material[value.upper()]
    except KeyError:
        return Material.MISC


def _parse_category(value: str) -> ObjectCategory:
    """Parse category string to ObjectCategory enum."""
    if value is None:
        return ObjectCategory.MISC
    try:
        return ObjectCategory[value.upper()]
    except KeyError:
        return ObjectCategory.MISC


def _dict_to_physical(d: dict) -> PhysicalProperties:
    """Convert dict to PhysicalProperties."""
    if d is None:
        return PhysicalProperties()
    return PhysicalProperties(
        weight=d.get("weight", 0.0),
        width=d.get("width", 0.0),
        height=d.get("height", 0.0),
        depth=d.get("depth", 0.0),
        volume_capacity=d.get("volume_capacity", 0.0),
        material=_parse_material(d.get("material")),
    )


def _dict_to_state(d: dict) -> StateProperties:
    """Convert dict to StateProperties."""
    if d is None:
        return StateProperties()
    return StateProperties(
        is_open=d.get("is_open", False),
        is_on=d.get("is_on", False),
        is_full=d.get("is_full", False),
        fill_level=d.get("fill_level", 0.0),
        is_dirty=d.get("is_dirty", False),
        is_heated=d.get("is_heated", False),
        is_cooled=d.get("is_cooled", False),
        is_clean=d.get("is_clean", True),
    )


def _dict_to_affordance(d: dict) -> AffordanceProperties:
    """Convert dict to AffordanceProperties."""
    if d is None:
        return AffordanceProperties()
    return AffordanceProperties(
        can_contain=d.get("can_contain", False),
        can_support=d.get("can_support", False),
        can_cut=d.get("can_cut", False),
        can_heat=d.get("can_heat", False),
        can_cool=d.get("can_cool", False),
        can_pick=d.get("can_pick", False),
        can_open=d.get("can_open", False),
        can_close=d.get("can_close", False),
        can_toggle=d.get("can_toggle", False),
        can_slice=d.get("can_slice", False),
        can_fill=d.get("can_fill", False),
        can_pour=d.get("can_pour", False),
    )


def load_from_yaml(file_path: Union[str, Path]) -> KnowledgeBase:
    """
    Load a KnowledgeBase from a YAML file.

    Args:
        file_path: Path to YAML file containing object definitions

    Returns:
        KnowledgeBase populated with entities from the YAML file

    YAML format:
        objects:
          - id: "apple_1"
            name: "Apple"
            category: "FOOD"
            tags: ["fruit", "red"]
            physical:
              weight: 180.0
              material: "FOOD"
            state:
              is_clean: true
            affordance:
              can_pick: true
              can_contain: true
              can_slice: true
    """
    kb = KnowledgeBase()
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    objects_list = data.get("objects", [])

    for obj_data in objects_list:
        entity = ObjectEntity(
            id=obj_data["id"],
            name=obj_data["name"],
            category=_parse_category(obj_data.get("category")),
            tags=obj_data.get("tags", []),
            physical=_dict_to_physical(obj_data.get("physical")),
            state=_dict_to_state(obj_data.get("state")),
            affordance=_dict_to_affordance(obj_data.get("affordance")),
        )
        kb.add(entity)

    return kb


def save_to_yaml(kb: KnowledgeBase, file_path: Union[str, Path]) -> None:
    """
    Save a KnowledgeBase to a YAML file.

    Args:
        kb: KnowledgeBase to save
        file_path: Output YAML file path
    """
    objects_list = []

    for obj in kb.get_all():
        obj_dict = {
            "id": obj.id,
            "name": obj.name,
            "category": obj.category.name,
            "tags": obj.tags,
            "physical": {
                "weight": obj.physical.weight,
                "width": obj.physical.width,
                "height": obj.physical.height,
                "depth": obj.physical.depth,
                "volume_capacity": obj.physical.volume_capacity,
                "material": obj.physical.material.name,
            },
            "state": {
                "is_open": obj.state.is_open,
                "is_on": obj.state.is_on,
                "is_full": obj.state.is_full,
                "fill_level": obj.state.fill_level,
                "is_dirty": obj.state.is_dirty,
                "is_heated": obj.state.is_heated,
                "is_cooled": obj.state.is_cooled,
                "is_clean": obj.state.is_clean,
            },
            "affordance": {
                "can_contain": obj.affordance.can_contain,
                "can_support": obj.affordance.can_support,
                "can_cut": obj.affordance.can_cut,
                "can_heat": obj.affordance.can_heat,
                "can_cool": obj.affordance.can_cool,
                "can_pick": obj.affordance.can_pick,
                "can_open": obj.affordance.can_open,
                "can_close": obj.affordance.can_close,
                "can_toggle": obj.affordance.can_toggle,
                "can_slice": obj.affordance.can_slice,
                "can_fill": obj.affordance.can_fill,
                "can_pour": obj.affordance.can_pour,
            },
        }
        objects_list.append(obj_dict)

    output_data = {"objects": objects_list}

    with open(file_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
