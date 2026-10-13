from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set


@dataclass
class Observation:
    """Represents the raw sensory feedback returned by the environment to the agent."""
    current_room: str
    visible_items: List[str]
    doors: Dict[str, bool]  # destination_room -> is_locked
    inventory: List[str]
    message: str
    success: bool = True


class MultiRoomEnvironment:
    """
    Discrete multi-room environment consisting of Room A and Room B connected by a door.
    
    Structure:
      Room A: Contains 'key', 'door_to_B' (locked), 'box'
      Room B: Contains 'target_pedestal', 'obstacle'
      
    Task: Transport 'box' from Room A to Room B and place it on 'target_pedestal'.
    """

    def __init__(self):
        # Room contents
        self.rooms: Dict[str, Set[str]] = {
            "Room A": {"key", "box"},
            "Room B": {"target_pedestal"}
        }
        # Door states: (from_room, to_room) -> is_locked
        self.doors: Dict[tuple, bool] = {
            ("Room A", "Room B"): True,
            ("Room B", "Room A"): False
        }
        # Agent state
        self.agent_room: str = "Room A"
        self.agent_inventory: List[str] = []
        self.max_inventory_capacity: int = 1
        self.step_count: int = 0

    def get_observation(self, status_msg: str = "", success: bool = True) -> Observation:
        """Constructs an observation slice based on agent's current position."""
        doors_info = {}
        for (src, dst), is_locked in self.doors.items():
            if src == self.agent_room:
                doors_info[dst] = is_locked

        return Observation(
            current_room=self.agent_room,
            visible_items=sorted(list(self.rooms[self.agent_room])),
            doors=doors_info,
            inventory=list(self.agent_inventory),
            message=status_msg,
            success=success
        )

    def move_to(self, target_room: str) -> Observation:
        """Transitions agent to an adjacent room if connection exists and is unlocked."""
        self.step_count += 1
        door_key = (self.agent_room, target_room)

        if door_key not in self.doors:
            return self.get_observation(
                f"Movement failed: No direct passage from {self.agent_room} to {target_room}.",
                success=False
            )

        if self.doors[door_key]:
            return self.get_observation(
                f"Movement failed: Door leading to {target_room} is locked.",
                success=False
            )

        self.agent_room = target_room
        return self.get_observation(f"Moved successfully to {target_room}.")

    def pickup(self, item_name: str) -> Observation:
        """Transfers an item from the current room into the agent's inventory."""
        self.step_count += 1
        if len(self.agent_inventory) >= self.max_inventory_capacity:
            return self.get_observation(
                f"Pickup failed: Inventory full (carrying {self.agent_inventory}).",
                success=False
            )

        if item_name not in self.rooms[self.agent_room]:
            return self.get_observation(
                f"Pickup failed: Item '{item_name}' is not in {self.agent_room}.",
                success=False
            )

        self.rooms[self.agent_room].remove(item_name)
        self.agent_inventory.append(item_name)
        return self.get_observation(f"Picked up '{item_name}'.")

    def drop(self, item_name: str) -> Observation:
        """Transfers an item from inventory to current room floor."""
        self.step_count += 1
        if item_name not in self.agent_inventory:
            return self.get_observation(
                f"Drop failed: Agent does not possess '{item_name}'.",
                success=False
            )

        self.agent_inventory.remove(item_name)
        self.rooms[self.agent_room].add(item_name)
        return self.get_observation(f"Dropped '{item_name}' into {self.agent_room}.")

    def open_door(self, target_room: str) -> Observation:
        """Unlocks door connecting current room to target room using required key."""
        self.step_count += 1
        door_key = (self.agent_room, target_room)

        if door_key not in self.doors:
            return self.get_observation(
                f"Open failed: No door to {target_room}.",
                success=False
            )

        if not self.doors[door_key]:
            return self.get_observation(f"Door to {target_room} is already unlocked.")

        if "key" not in self.agent_inventory:
            return self.get_observation(
                f"Open failed: You need 'key' in inventory to unlock door to {target_room}.",
                success=False
            )

        self.doors[door_key] = False
        return self.get_observation(f"Door to {target_room} unlocked successfully!")

    def is_goal_satisfied(self, target_item: str = "box", target_location: str = "Room B") -> bool:
        """Checks if the end condition is met."""
        return target_item in self.rooms[target_location]