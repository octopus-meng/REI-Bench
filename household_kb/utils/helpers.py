"""Utility functions for the household knowledge base."""

from typing import List
from entities import ObjectEntity, ObjectCategory, Material, PhysicalProperties, AffordanceProperties, StateProperties


def create_food_object(
    obj_id: str,
    name: str,
    weight: float = 0.0,
    tags: List[str] = None,
) -> ObjectEntity:
    """Helper to create a food category object."""
    return ObjectEntity(
        id=obj_id,
        name=name,
        category=ObjectCategory.FOOD,
        tags=tags or [],
        physical=PhysicalProperties(weight=weight, material=Material.FOOD),
        affordance=AffordanceProperties(can_pick=True, can_contain=True),
    )


def create_container_object(
    obj_id: str,
    name: str,
    capacity: float = 0.0,
    can_open: bool = False,
    can_heat: bool = False,
    can_cool: bool = False,
    can_support: bool = False,
    tags: List[str] = None,
) -> ObjectEntity:
    """Helper to create a container category object."""
    return ObjectEntity(
        id=obj_id,
        name=name,
        category=ObjectCategory.CONTAINER,
        tags=tags or [],
        physical=PhysicalProperties(volume_capacity=capacity),
        affordance=AffordanceProperties(
            can_contain=True,
            can_open=can_open,
            can_heat=can_heat,
            can_cool=can_cool,
            can_support=can_support,
        ),
    )


def create_appliance_object(
    obj_id: str,
    name: str,
    can_toggle: bool = True,
    can_open: bool = False,
    can_heat: bool = False,
    tags: List[str] = None,
) -> ObjectEntity:
    """Helper to create an appliance category object."""
    return ObjectEntity(
        id=obj_id,
        name=name,
        category=ObjectCategory.APPLIANCE,
        tags=tags or [],
        affordance=AffordanceProperties(
            can_toggle=can_toggle,
            can_open=can_open,
            can_heat=can_heat,
        ),
    )


def create_utensil_object(
    obj_id: str,
    name: str,
    can_cut: bool = False,
    can_support: bool = False,
    tags: List[str] = None,
) -> ObjectEntity:
    """Helper to create a utensil category object."""
    return ObjectEntity(
        id=obj_id,
        name=name,
        category=ObjectCategory.UTENSIL,
        tags=tags or [],
        affordance=AffordanceProperties(
            can_pick=True,
            can_cut=can_cut,
            can_support=can_support,
        ),
    )
