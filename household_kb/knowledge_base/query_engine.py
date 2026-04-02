"""Query engine for filtering and searching the knowledge base."""

from typing import Callable, Optional, List
from dataclasses import dataclass
from entities import ObjectEntity, ObjectCategory, Material


@dataclass
class QueryCriteria:
    """Criteria for filtering objects in the knowledge base."""
    category: Optional[ObjectCategory] = None
    material: Optional[Material] = None
    can_pick: Optional[bool] = None
    can_contain: Optional[bool] = None
    can_support: Optional[bool] = None
    can_cut: Optional[bool] = None
    can_slice: Optional[bool] = None
    can_heat: Optional[bool] = None
    can_cool: Optional[bool] = None
    can_open: Optional[bool] = None
    can_toggle: Optional[bool] = None
    is_open: Optional[bool] = None
    is_on: Optional[bool] = None
    is_full: Optional[bool] = None
    tags: Optional[List[str]] = None
    name_contains: Optional[str] = None
    custom_filter: Optional[Callable[[ObjectEntity], bool]] = None


class QueryEngine:
    """Handles filtering and querying of objects."""

    def filter(self, objects: List[ObjectEntity], criteria: QueryCriteria) -> List[ObjectEntity]:
        """
        Apply query criteria to filter objects.

        Args:
            objects: List of ObjectEntity to filter
            criteria: QueryCriteria defining the filter conditions

        Returns:
            Filtered list of ObjectEntity matching all criteria
        """
        results = objects

        if criteria.category is not None:
            results = [o for o in results if o.category == criteria.category]

        if criteria.material is not None:
            results = [o for o in results if o.physical.material == criteria.material]

        if criteria.can_pick is not None:
            results = [o for o in results if o.affordance.can_pick == criteria.can_pick]

        if criteria.can_contain is not None:
            results = [o for o in results if o.affordance.can_contain == criteria.can_contain]

        if criteria.can_support is not None:
            results = [o for o in results if o.affordance.can_support == criteria.can_support]

        if criteria.can_cut is not None:
            results = [o for o in results if o.affordance.can_cut == criteria.can_cut]

        if criteria.can_slice is not None:
            results = [o for o in results if o.affordance.can_slice == criteria.can_slice]

        if criteria.can_heat is not None:
            results = [o for o in results if o.affordance.can_heat == criteria.can_heat]

        if criteria.can_cool is not None:
            results = [o for o in results if o.affordance.can_cool == criteria.can_cool]

        if criteria.can_open is not None:
            results = [o for o in results if o.affordance.can_open == criteria.can_open]

        if criteria.can_toggle is not None:
            results = [o for o in results if o.affordance.can_toggle == criteria.can_toggle]

        if criteria.is_open is not None:
            results = [o for o in results if o.state.is_open == criteria.is_open]

        if criteria.is_on is not None:
            results = [o for o in results if o.state.is_on == criteria.is_on]

        if criteria.is_full is not None:
            results = [o for o in results if o.state.is_full == criteria.is_full]

        if criteria.tags:
            results = [
                o for o in results
                if any(tag.lower() in [t.lower() for t in o.tags] for tag in criteria.tags)
            ]

        if criteria.name_contains is not None:
            results = [
                o for o in results
                if criteria.name_contains.lower() in o.name.lower()
            ]

        if criteria.custom_filter is not None:
            results = [o for o in results if criteria.custom_filter(o)]

        return results
