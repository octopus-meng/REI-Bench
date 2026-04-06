"""KnowledgeBase for storing and querying household object entities."""

from typing import Optional, List, Dict
from entities import ObjectEntity, ObjectCategory
from .query_engine import QueryEngine, QueryCriteria


class KnowledgeBase:
    """
    A knowledge base for household embodied agents.

    Provides storage and querying capabilities for ObjectEntity objects,
    supporting addition, lookup, and filtered queries.

    Example:
        kb = KnowledgeBase()
        kb.add(apple)
        kb.add(fridge)

        # Query all pickable objects
        pickable = kb.query_subclass_by_affordance(can_pick=True)

        # Filter by category
        fruits = kb.filter(QueryCriteria(category=ObjectCategory.FOOD))

        # Query entity attributes
        weights = kb.query_entities_attr(fruits, "physical.weight")
    """

    def __init__(self):
        self._objects: Dict[str, ObjectEntity] = {}
        self._query_engine = QueryEngine()

    def add(self, entity: ObjectEntity) -> None:
        """
        Add an object entity to the knowledge base.

        Args:
            entity: ObjectEntity to add

        Raises:
            ValueError: If an object with the same id already exists
        """
        if entity.id in self._objects:
            raise ValueError(f"Object with id '{entity.id}' already exists")
        self._objects[entity.id] = entity

    def remove(self, entity_id: str) -> bool:
        """
        Remove an object by its id.

        Args:
            entity_id: Unique identifier of the object

        Returns:
            True if removed, False if not found
        """
        if entity_id in self._objects:
            del self._objects[entity_id]
            return True
        return False

    def get(self, entity_id: str) -> Optional[ObjectEntity]:
        """
        Retrieve an object by its id.

        Args:
            entity_id: Unique identifier of the object

        Returns:
            ObjectEntity if found, None otherwise
        """
        return self._objects.get(entity_id)

    def get_all(self) -> List[ObjectEntity]:
        """Return all objects in the knowledge base."""
        return list(self._objects.values())

    def query_subclass_by_category(self, category: ObjectCategory) -> List[ObjectEntity]:
        """
        Query all objects belonging to a specific category.

        Args:
            category: ObjectCategory to filter by

        Returns:
            List of ObjectEntity matching the category
        """
        return [obj for obj in self._objects.values() if obj.category == category]

    def query_subclass_by_affordance(
        self,
        can_pick: Optional[bool] = None,
        can_contain: Optional[bool] = None,
        can_support: Optional[bool] = None,
        can_cut: Optional[bool] = None,
        can_heat: Optional[bool] = None,
        can_cool: Optional[bool] = None,
        can_open: Optional[bool] = None,
        can_toggle: Optional[bool] = None,
    ) -> List[ObjectEntity]:
        """
        Query objects by affordance properties.

        Args:
            can_pick: Filter for pickable objects
            can_contain: Filter for container objects
            can_support: Filter for support objects
            can_cut: Filter for cuttable objects
            can_heat: Filter for heatable objects
            can_cool: Filter for coolable objects
            can_open: Filter for openable objects
            can_toggle: Filter for toggleable objects

        Returns:
            List of ObjectEntity matching the affordance criteria
        """
        criteria = QueryCriteria(
            can_pick=can_pick,
            can_contain=can_contain,
            can_support=can_support,
            can_cut=can_cut,
            can_heat=can_heat,
            can_cool=can_cool,
            can_open=can_open,
            can_toggle=can_toggle,
        )
        return self.filter(criteria)

    def filter(self, criteria: QueryCriteria) -> List[ObjectEntity]:
        """
        Filter objects using detailed QueryCriteria.

        Args:
            criteria: QueryCriteria specifying filter conditions

        Returns:
            List of ObjectEntity matching all criteria
        """
        return self._query_engine.filter(list(self._objects.values()), criteria)

    def resolve_vague_reference(self, reference: str) -> List[ObjectEntity]:
        """
        Resolve a vague reference (e.g., 'fruit', 'container') to concrete objects.

        Args:
            reference: Vague descriptor string

        Returns:
            List of ObjectEntity that match the vague reference
        """
        return [
            obj for obj in self._objects.values()
            if obj.matches_vague_reference(reference)
        ]

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._objects

    def query_entities_attr(self, entities: List[ObjectEntity], attr_path: str) -> List[tuple]:
        """
        Query specific attributes from a list of entities.

        Args:
            entities: List of ObjectEntity to query from
            attr_path: Dot-separated attribute path (e.g., "physical.weight", "state.is_open")
                      Supports top-level attrs: id, name, category, tags
                      Supports nested attrs: physical.*, state.*, affordance.*

        Returns:
            List of (ObjectEntity, value) tuples for each entity with valid attribute

        Examples:
            kb.query_entities_attr([apple, banana], "physical.weight")
            kb.query_entities_attr(kb.get_all(), "state.is_open")
            kb.query_entities_attr(results, "physical.material")
        """
        if entities is None:
            entities = list(self._objects.keys())
        results = []
        for entity in entities:
            value = entity.get_attr(attr_path)
            if value is not None:
                results.append((entity, value))
        return results
