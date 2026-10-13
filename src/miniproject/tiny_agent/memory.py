from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from miniproject.tiny_agent.environment import Observation


@dataclass
class MemoryRecord:
    """A single episodic step recording thought, action, and resulting observation."""
    step: int
    thought: str
    action_tool: str
    action_args: Dict[str, Any]
    observation_summary: str
    success: bool


class AgentMemory:
    """
    Working Memory & Belief State for TinyAgent.
    
    Responsibilities:
      1. Episodic Trace: History of reasoning, chosen actions, and outcomes.
      2. Mental Map: Rooms visited, known door connections, lock statuses.
      3. Item Beliefs: Where objects were last observed or if held in inventory.
    """

    def __init__(self):
        self.history: List[MemoryRecord] = []
        self.current_location: str = "Unknown"
        self.inventory: Set[str] = set()
        
        # Spatial beliefs
        self.visited_rooms: Set[str] = set()
        self.known_doors: Dict[str, Dict[str, bool]] = {}  # room -> {target: is_locked}
        self.known_item_locations: Dict[str, str] = {}    # item -> room

    def update_from_observation(self, obs: Observation, step: int, thought: str, tool: str, args: Dict[str, Any]) -> None:
        """Integrates newly sensed information into long-term belief state."""
        self.current_location = obs.current_room
        self.visited_rooms.add(obs.current_room)
        self.inventory = set(obs.inventory)

        # Update doors state
        if obs.current_room not in self.known_doors:
            self.known_doors[obs.current_room] = {}
        for target, locked in obs.doors.items():
            self.known_doors[obs.current_room][target] = locked

        # Update items in room
        for item in obs.visible_items:
            self.known_item_locations[item] = obs.current_room

        # If an item is in inventory, remove room location mapping
        for inv_item in obs.inventory:
            if inv_item in self.known_item_locations:
                del self.known_item_locations[inv_item]

        # Log episodic trace
        record = MemoryRecord(
            step=step,
            thought=thought,
            action_tool=tool,
            action_args=args,
            observation_summary=obs.message or f"Observed {obs.current_room}",
            success=obs.success
        )
        self.history.append(record)

    def get_state_summary(self) -> str:
        """Returns a concise representation of the current internal belief state."""
        items_map = ", ".join([f"{k} in {v}" for k, v in self.known_item_locations.items()]) or "None"
        inv_str = ", ".join(self.inventory) if self.inventory else "Empty"
        
        return (
            f"Agent Location: {self.current_location} | "
            f"Inventory: [{inv_str}] | "
            f"Known Items: [{items_map}] | "
            f"Visited Rooms: {sorted(list(self.visited_rooms))}"
        )