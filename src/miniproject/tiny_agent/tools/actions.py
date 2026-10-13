from miniproject.tiny_agent.tools.base import BaseTool, ToolResult
from miniproject.tiny_agent.environment import MultiRoomEnvironment


class LookTool(BaseTool):
    """Inspects the current room, visible items, and adjacent doors."""

    def __init__(self, env: MultiRoomEnvironment):
        super().__init__(
            name="look",
            description="Inspect surroundings, items in current room, and connected doors.",
            usage_doc="look()"
        )
        self.env = env

    def execute(self, **kwargs) -> ToolResult:
        obs = self.env.get_observation()
        doors_status = ", ".join([f"{r} (locked={lock})" for r, lock in obs.doors.items()]) or "None"
        items_str = ", ".join(obs.visible_items) if obs.visible_items else "empty"
        inv_str = ", ".join(obs.inventory) if obs.inventory else "empty"

        report = (
            f"Location: {obs.current_room} | "
            f"Visible items: [{items_str}] | "
            f"Doors: [{doors_status}] | "
            f"Inventory: [{inv_str}]"
        )
        return ToolResult(success=True, output=report, metadata={"observation": obs})


class MoveTool(BaseTool):
    """Moves the agent to an adjacent connected room."""

    def __init__(self, env: MultiRoomEnvironment):
        super().__init__(
            name="move",
            description="Move agent to a specified destination room.",
            usage_doc="move(target_room='Room B')"
        )
        self.env = env

    def execute(self, target_room: str = "", **kwargs) -> ToolResult:
        if not target_room:
            return ToolResult(success=False, output="Parameter 'target_room' is required.")
        obs = self.env.move_to(target_room)
        return ToolResult(success=obs.success, output=obs.message, metadata={"observation": obs})


class PickUpTool(BaseTool):
    """Picks up an item from the current room."""

    def __init__(self, env: MultiRoomEnvironment):
        super().__init__(
            name="pickup",
            description="Pick up an item located in the current room into inventory.",
            usage_doc="pickup(item='key')"
        )
        self.env = env

    def execute(self, item: str = "", **kwargs) -> ToolResult:
        if not item:
            return ToolResult(success=False, output="Parameter 'item' is required.")
        obs = self.env.pickup(item)
        return ToolResult(success=obs.success, output=obs.message, metadata={"observation": obs})


class DropTool(BaseTool):
    """Drops an item from inventory into current room."""

    def __init__(self, env: MultiRoomEnvironment):
        super().__init__(
            name="drop",
            description="Drop an item currently held in inventory into the current room.",
            usage_doc="drop(item='box')"
        )
        self.env = env

    def execute(self, item: str = "", **kwargs) -> ToolResult:
        if not item:
            return ToolResult(success=False, output="Parameter 'item' is required.")
        obs = self.env.drop(item)
        return ToolResult(success=obs.success, output=obs.message, metadata={"observation": obs})


class OpenTool(BaseTool):
    """Opens/unlocks a door leading to a specified room."""

    def __init__(self, env: MultiRoomEnvironment):
        super().__init__(
            name="open",
            description="Unlock a locked door to target room (requires appropriate key in inventory).",
            usage_doc="open(target_room='Room B')"
        )
        self.env = env

    def execute(self, target_room: str = "", **kwargs) -> ToolResult:
        if not target_room:
            return ToolResult(success=False, output="Parameter 'target_room' is required.")
        obs = self.env.open_door(target_room)
        return ToolResult(success=obs.success, output=obs.message, metadata={"observation": obs})