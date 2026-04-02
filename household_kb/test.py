#!/usr/bin/env python3
"""Example usage of the Household Knowledge Base."""

from entities import (
    ObjectEntity, ObjectCategory, Material,
    PhysicalProperties, StateProperties, AffordanceProperties
)
from entities.enums import ContainmentState
from knowledge_base import KnowledgeBase, QueryCriteria, load_from_yaml, save_to_yaml
from utils import (
    create_food_object,
    create_container_object,
    create_appliance_object,
    create_utensil_object,
)


def demo_yaml_loading():
    """Demonstrate loading knowledge base from YAML file."""
    print("\n" + "=" * 50)
    print("YAML Loading Demo")
    print("=" * 50)

    # Load ALFRED objects from YAML
    yaml_path = "data/alfred_objects_new.yaml"
    kb_alfred = load_from_yaml(yaml_path)

    print(f"\nLoaded {len(kb_alfred)} objects from YAML")

    # Query all pickable objects
    print("\n--- Pickable Objects (first 10) ---")
    pickable = kb_alfred.query_subclass_by_affordance(can_pick=True)
    for obj in pickable[:10]:
        print(f"  - {obj.name}")
    print(f"  ... and {len(pickable) - 10} more")

    # Query all sliceable objects
    print("\n--- Sliceable Objects ---")
    for obj in kb_alfred.filter(QueryCriteria(can_slice=True)):
        print(f"  - {obj.name}")

    # Query all openable objects
    print("\n--- Openable Objects ---")
    for obj in kb_alfred.filter(QueryCriteria(can_open=True)):
        print(f"  - {obj.name}")

    # Query all toggleable objects
    print("\n--- Toggleable Objects ---")
    for obj in kb_alfred.filter(QueryCriteria(can_toggle=True)):
        print(f"  - {obj.name}")

    # Query by category
    print("\n--- Appliances ---")
    for obj in kb_alfred.query_subclass_by_category(ObjectCategory.APPLIANCE):
        print(f"  - {obj.name}")

    # Resolve vague references
    print("\n--- Vague Reference: 'kitchen' ---")
    for obj in kb_alfred.resolve_vague_reference("kitchen"):
        print(f"  - {obj.name}")

    print("\n--- Vague Reference: 'fruit' ---")
    for obj in kb_alfred.resolve_vague_reference("fruit"):
        print(f"  - {obj.name}")

    return kb_alfred


def main():
    # Demo 1: Programmatic construction
    print("=" * 50)
    print("Programmatic Construction Demo")
    print("=" * 50)

    # Initialize knowledge base
    kb = KnowledgeBase()

    # Add some objects using helpers
    kb.add(create_food_object("apple_1", "Apple", weight=180.0, tags=["fruit"]))
    kb.add(create_food_object("bread_1", "Bread", weight=400.0, tags=["bakery"]))
    kb.add(create_food_object("egg_1", "Egg", weight=50.0, tags=["protein"]))

    kb.add(create_container_object("bowl_1", "Bowl", capacity=500.0, can_support=True))
    kb.add(create_container_object("fridge_1", "Fridge", capacity=10000.0, can_open=True, can_cool=True))
    kb.add(create_container_object("microwave_1", "Microwave", capacity=2000.0, can_open=True, can_heat=True))

    kb.add(create_appliance_object("faucet_1", "Faucet", can_toggle=True))
    kb.add(create_appliance_object("stove_1", "Stove", can_toggle=True, can_heat=True))

    kb.add(create_utensil_object("knife_1", "Knife", can_cut=True))
    kb.add(create_utensil_object("spoon_1", "Spoon", can_support=True))
    kb.add(create_utensil_object("plate_1", "Plate", can_support=True))

    # Add a custom object with full specification
    kb.add(ObjectEntity(
        id="tomato_1",
        name="Tomato",
        category=ObjectCategory.FOOD,
        tags=["vegetable", "red", "sliceable"],
        physical=PhysicalProperties(weight=120.0, material=Material.FOOD),
        state=StateProperties(is_dirty=False, is_clean=True),
        affordance=AffordanceProperties(
            can_pick=True,
            can_contain=True,
            can_cut=True,
            can_slice=True,
        ),
    ))

    print(f"Knowledge base contains {len(kb)} objects\n")

    # Query: all pickable objects
    print("=== Pickable Objects ===")
    for obj in kb.query_subclass_by_affordance(can_pick=True):
        print(f"  - {obj.name} (id: {obj.id})")

    # Query: all containers
    print("\n=== Containers ===")
    for obj in kb.query_subclass_by_affordance(can_contain=True):
        print(f"  - {obj.name} (capacity: {obj.physical.volume_capacity}ml)")

    # Query: objects that can be heated
    print("\n=== Heatable Objects ===")
    for obj in kb.query_subclass_by_affordance(can_heat=True):
        print(f"  - {obj.name}")

    # Query: objects that can be sliced
    print("\n=== Sliceable Objects ===")
    for obj in kb.filter(QueryCriteria(can_slice=True)):
        print(f"  - {obj.name}")

    # Resolve vague reference
    print("\n=== Vague Reference: 'fruit' ===")
    for obj in kb.resolve_vague_reference("fruit"):
        print(f"  - {obj.name}")

    print("\n=== Vague Reference: 'electronic' ===")
    # None exist in current KB
    for obj in kb.resolve_vague_reference("electronic"):
        print(f"  - {obj.name}")

    # Filter by custom criteria
    print("\n=== Custom Filter: objects with 'red' tag ===")
    results = kb.filter(QueryCriteria(
        custom_filter=lambda o: "red" in [t.lower() for t in o.tags]
    ))
    for obj in results:
        print(f"  - {obj.name}")

    # Demo 2: YAML loading
    demo_yaml_loading()


if __name__ == "__main__":
    main()
