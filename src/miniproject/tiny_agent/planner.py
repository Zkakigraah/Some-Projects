from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from miniproject.tiny_agent.memory import AgentMemory


@dataclass
class PlannedAction:
    """An action determined by the reasoning module to be performed next."""
    thought: str
    tool_name: str
    arguments: Dict[str, Any]
    is_terminal: bool = False


class BasePlanner(ABC):
    """
    Abstract interface for reasoning.
    Takes current goal and memory state, and generates the next action.
    Can be implemented via Rule-Based logic, classical planning (STRIPS/A*), or an LLM.
    """

    @abstractmethod
    def plan_next_step(self, goal: Dict[str, str], memory: AgentMemory) -> PlannedAction:
        """Determine next action given the goal and memory."""
        pass


class GoalDirectedPlanner(BasePlanner):
    """
    Symbolic state-machine planner that solves the transfer task:
    Move target_item to target_room.
    
    Reasoning Chain:
      1. If environment has not been inspected yet -> 'look'
      2. If target item in inventory & agent in target_room -> 'drop target_item'
      3. If target item in inventory & agent not in target_room:
           - If door to target is locked -> need key
           - If door unlocked -> 'move to target_room'
      4. If target item not in inventory:
           - If target item location unknown -> explore / 'look'
           - If key is required to reach target room and door is locked:
               * If key not in inventory and key location known -> 'pickup key'
               * If key in inventory and door locked -> 'open target_room'
               * If door unlocked and key in inventory -> drop key or move
           - If door is open & target item in current room -> 'pickup target_item'
    """

    def plan_next_step(self, goal: Dict[str, str], memory: AgentMemory) -> PlannedAction:
        target_item = goal.get("item", "box")
        target_room = goal.get("target_room", "Room B")
        current_loc = memory.current_location

        # Rule 0: Unknown surroundings -> Inspect first
        if current_loc == "Unknown" or not memory.history:
            return PlannedAction(
                thought="I have just initialized and don't know my surroundings. I must observe first.",
                tool_name="look",
                arguments={}
            )

        # Rule 1: Goal satisfaction check
        if memory.known_item_locations.get(target_item) == target_room and target_item not in memory.inventory:
            return PlannedAction(
                thought=f"Goal achieved: '{target_item}' is positioned safely in '{target_room}'!",
                tool_name="look",
                arguments={},
                is_terminal=True
            )

        # Rule 2: If holding target item and already at target room -> Drop it
        if target_item in memory.inventory:
            if current_loc == target_room:
                return PlannedAction(
                    thought=f"I have arrived at {target_room} with {target_item}. Dropping it to complete goal.",
                    tool_name="drop",
                    arguments={"item": target_item}
                )
            else:
                # Need to move to target room
                return PlannedAction(
                    thought=f"I am holding {target_item}. Moving into destination room {target_room}.",
                    tool_name="move",
                    arguments={"target_room": target_room}
                )

        # Rule 3: Door to target room is locked; we must unlock it
        doors_from_current = memory.known_doors.get(current_loc, {})
        is_door_locked = doors_from_current.get(target_room, False)

        if is_door_locked:
            if "key" in memory.inventory:
                return PlannedAction(
                    thought=f"The door to {target_room} is locked, and I possess the 'key'. Unlocking the door.",
                    tool_name="open",
                    arguments={"target_room": target_room}
                )
            elif "key" in memory.known_item_locations:
                return PlannedAction(
                    thought=f"Door to {target_room} is locked. I know 'key' is in {current_loc}. Picking up key.",
                    tool_name="pickup",
                    arguments={"item": "key"}
                )

        # Rule 4: If key is held but door is already unlocked, drop key to free inventory capacity
        if "key" in memory.inventory and not is_door_locked:
            return PlannedAction(
                thought="Door is unlocked. Dropping key to free inventory capacity for the target item.",
                tool_name="drop",
                arguments={"item": "key"}
            )

        # Rule 5: Pickup target item if present in current room
        if memory.known_item_locations.get(target_item) == current_loc:
            return PlannedAction(
                thought=f"The path will soon be clear and '{target_item}' is here. Picking up '{target_item}'.",
                tool_name="pickup",
                arguments={"item": target_item}
            )

        # Rule 6: Fallback look or explore
        return PlannedAction(
            thought="Gathering updated observation to revise plan.",
            tool_name="look",
            arguments={}
        )


import json
import re
import urllib.request
import urllib.error


class LLMPlanner(BasePlanner):
    """
    LLM-based reasoning engine implementing the ReAct (Reasoning + Acting) pattern.
    
    Generates thoughts and tool calls in the format:
      Thought: <natural language reasoning about state and goal>
      Action: <tool_name or 'finish'>
      Action Input: <json object or key=value dictionary of arguments>
      
    Supports:
      - Live OpenAI / Gemini / Ollama / DeepSeek compatible REST APIs
      - Built-in dynamic Mock Engine (zero-cost offline execution for experiments)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "mock-gpt-4o"
    ):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1/chat/completions"
        self.model_name = model_name

    def plan_next_step(self, goal: Dict[str, str], memory: AgentMemory) -> PlannedAction:
        """Constructs prompt, queries the LLM, and parses structured ReAct output."""
        prompt = self._build_react_prompt(goal, memory)
        raw_response = self._call_llm(prompt, memory, goal)
        return self._parse_react_response(raw_response)

    def _build_react_prompt(self, goal: Dict[str, str], memory: AgentMemory) -> str:
        """Constructs prompt providing tool specs, internal beliefs, and objective."""
        return f"""You are an autonomous agent operating in a multi-room environment.
Your Goal: Move '{goal.get('item', 'box')}' to '{goal.get('target_room', 'Room B')}'.

Available Tools:
- look(): Inspect surroundings
- move(target_room): Walk to adjacent unlocked room
- pickup(item): Pick up item from current room (Max inventory capacity = 1)
- drop(item): Drop item from inventory into current room
- open(target_room): Unlock door to room using key

Current Memory & Belief State:
- Location: {memory.current_location}
- Inventory: {list(memory.inventory)}
- Known Item Positions: {memory.known_item_locations}
- Known Doors: {memory.known_doors.get(memory.current_location, {})}

Answer STRICTLY in this ReAct format:
Thought: <what you think and why>
Action: <one of: look, move, pickup, drop, open, finish>
Action Input: {{"param": "value"}}
"""

    def _call_llm(self, prompt: str, memory: AgentMemory, goal: Dict[str, str]) -> str:
        """Dispatches prompt to live API endpoint or simulated cognitive mock."""
        # 1. Live API Mode (if valid key is provided)
        if self.api_key and self.api_key != "mock":
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                payload = json.dumps({
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a precise, reasoning-first AI agent."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }).encode("utf-8")

                req = urllib.request.Request(self.base_url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"⚠️ [LLM API Warning]: Falling back to local cognitive simulator ({e})")

        # 2. Local Cognitive Simulation Mode (runs out-of-the-box offline)
        return self._simulate_llm_cognition(memory, goal)

    def _simulate_llm_cognition(self, memory: AgentMemory, goal: Dict[str, str]) -> str:
        """Simulates how a frontier LLM reasons step-by-step using in-context prompt."""
        target_item = goal.get("item", "box")
        target_room = goal.get("target_room", "Room B")
        loc = memory.current_location
        inv = memory.inventory
        doors = memory.known_doors.get(loc, {})
        is_locked = doors.get(target_room, False)

        if memory.known_item_locations.get(target_item) == target_room and target_item not in inv:
            return (
                f"Thought: I observe '{target_item}' is now placed inside '{target_room}'. Mission objective is fulfilled.\n"
                f"Action: finish\n"
                f"Action Input: {{}}"
            )

        if target_item in inv:
            if loc == target_room:
                return (
                    f"Thought: I have arrived at '{target_room}' carrying '{target_item}'. I need to place it down.\n"
                    f"Action: drop\n"
                    f"Action Input: {{\"item\": \"{target_item}\"}}"
                )
            else:
                return (
                    f"Thought: I am carrying '{target_item}'. Now moving to destination '{target_room}'.\n"
                    f"Action: move\n"
                    f"Action Input: {{\"target_room\": \"{target_room}\"}}"
                )

        if is_locked:
            if "key" in inv:
                return (
                    f"Thought: The door to '{target_room}' is locked and I am holding the key. I will unlock it now.\n"
                    f"Action: open\n"
                    f"Action Input: {{\"target_room\": \"{target_room}\"}}"
                )
            else:
                return (
                    f"Thought: Door to '{target_room}' is locked. I need the key from '{loc}' first.\n"
                    f"Action: pickup\n"
                    f"Action Input: {{\"item\": \"key\"}}"
                )

        if "key" in inv and not is_locked:
            return (
                f"Thought: Door is open, but my inventory is full with 'key'. I must drop it before taking '{target_item}'.\n"
                f"Action: drop\n"
                f"Action Input: {{\"item\": \"key\"}}"
            )

        return (
            f"Thought: The path is cleared and I have free inventory space. Picking up target '{target_item}'.\n"
            f"Action: pickup\n"
            f"Action Input: {{\"item\": \"{target_item}\"}}"
        )

    def _parse_react_response(self, text: str) -> PlannedAction:
        """Extracts Thought, Action, and Action Input fields from ReAct response."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*\}|[^\n]+)", text)

        thought = thought_match.group(1).strip() if thought_match else text.strip()
        tool_name = action_match.group(1).strip().lower() if action_match else "look"

        args = {}
        if input_match:
            raw_input = input_match.group(1).strip()
            try:
                args = json.loads(raw_input)
            except json.JSONDecodeError:
                pass

        if tool_name == "finish":
            return PlannedAction(
                thought=thought,
                tool_name="look",
                arguments={},
                is_terminal=True
            )

        return PlannedAction(
            thought=thought,
            tool_name=tool_name,
            arguments=args,
            is_terminal=False
        )