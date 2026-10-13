import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional

from miniproject.tiny_agent.planner import BasePlanner, PlannedAction
from miniproject.tiny_agent.memory import AgentMemory
from miniproject.tiny_agent.environment import MultiRoomEnvironment


class StateEncoder:
    """
    Transforms discrete AgentMemory & Goal into a fixed-length continuous feature vector.
    
    Feature Vector Layout (Dimension = 10):
      [0] Agent in Room A (1.0 or 0.0)
      [1] Agent in Room B (1.0 or 0.0)
      [2] Target Item ('box') in Room A
      [3] Target Item ('box') in Room B
      [4] Target Item in inventory
      [5] 'key' in Room A
      [6] 'key' in Room B
      [7] 'key' in inventory
      [8] Door (Room A -> Room B) is locked
      [9] Inventory is empty (capacity available)
    """

    FEATURE_DIM: int = 10

    @classmethod
    def encode(cls, memory: AgentMemory, goal: Dict[str, str]) -> torch.Tensor:
        target_item = goal.get("item", "box")
        vec = [0.0] * cls.FEATURE_DIM

        # Agent location
        if memory.current_location == "Room A":
            vec[0] = 1.0
        elif memory.current_location == "Room B":
            vec[1] = 1.0

        # Target item position
        box_loc = memory.known_item_locations.get(target_item)
        if box_loc == "Room A":
            vec[2] = 1.0
        elif box_loc == "Room B":
            vec[3] = 1.0
        if target_item in memory.inventory:
            vec[4] = 1.0

        # Key position
        key_loc = memory.known_item_locations.get("key")
        if key_loc == "Room A":
            vec[5] = 1.0
        elif key_loc == "Room B":
            vec[6] = 1.0
        if "key" in memory.inventory:
            vec[7] = 1.0

        # Door state from Room A to Room B
        doors_from_a = memory.known_doors.get("Room A", {})
        if doors_from_a.get("Room B", False):
            vec[8] = 1.0

        # Inventory empty flag
        if len(memory.inventory) == 0:
            vec[9] = 1.0

        return torch.tensor(vec, dtype=torch.float32)


class ActionCatalog:
    """
    Defines the discrete action output space of the Neural Policy.
    Maps an integer index [0..6] to an executable tool call and arguments.
    """

    ACTIONS: List[Dict[str, Any]] = [
        {"tool": "pickup", "args": {"item": "key"}, "desc": "Pick up key"},
        {"tool": "open", "args": {"target_room": "Room B"}, "desc": "Unlock door to Room B"},
        {"tool": "drop", "args": {"item": "key"}, "desc": "Drop key to clear inventory"},
        {"tool": "pickup", "args": {"item": "box"}, "desc": "Pick up target box"},
        {"tool": "move", "args": {"target_room": "Room B"}, "desc": "Move into Room B"},
        {"tool": "drop", "args": {"item": "box"}, "desc": "Drop target box at destination"},
        {"tool": "look", "args": {}, "desc": "Inspect surroundings"}
    ]

    NUM_ACTIONS: int = len(ACTIONS)

    @classmethod
    def get_action_details(cls, action_idx: int) -> Dict[str, Any]:
        if 0 <= action_idx < cls.NUM_ACTIONS:
            return cls.ACTIONS[action_idx]
        return cls.ACTIONS[-1]


class PolicyNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) mapping State Tensor -> Action Logits.
    π_θ(a | s) = softmax(MLP(s))
    """

    def __init__(self, state_dim: int = 10, hidden_dim: int = 64, num_actions: int = 7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Outputs unnormalized logits for each candidate action."""
        return self.network(state_tensor)


class PolicyTrainer:
    """
    Imitation Learning engine: Trains the neural policy using state-action pairs
    synthesized from optimal decision logic across varied configurations.
    """

    @classmethod
    def generate_expert_dataset(cls) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates diverse scenarios so the network learns generalized decision rules
        (e.g., if door unlocked, don't pick up key; if at destination with box, drop it).
        """
        states: List[List[float]] = []
        labels: List[int] = []

        # Scenario 1: Initial standard state (In Room A, door locked, key & box in A)
        # -> Action 0: pickup key
        states.append([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        labels.append(0)

        # Scenario 2: Holding key, door locked, in Room A
        # -> Action 1: open door to Room B
        states.append([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        labels.append(1)

        # Scenario 3: Door unlocked, holding key in Room A, box on floor
        # -> Action 2: drop key to free inventory
        states.append([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        labels.append(2)

        # Scenario 4: Door unlocked, inventory empty, in Room A, box in Room A
        # -> Action 3: pickup box
        states.append([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        labels.append(3)

        # Scenario 5: Holding box, door unlocked, in Room A
        # -> Action 4: move to Room B
        states.append([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        labels.append(4)

        # Scenario 6: Holding box, agent inside Room B
        # -> Action 5: drop box (Goal complete!)
        states.append([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        labels.append(5)

        # Scenario 7 (Generalization sample): Door starts already unlocked, key on floor
        # Neural net should directly pick up box (Action 3), skipping key entirely!
        states.append([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        labels.append(3)

        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    @classmethod
    def train(cls, model: PolicyNetwork, epochs: int = 150, lr: float = 0.005) -> None:
        """Performs supervised behavioral cloning training."""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        x_train, y_train = cls.generate_expert_dataset()

        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()


class NeuralPolicyPlanner(BasePlanner):
    """
    Substitutes symbolic rule-based planner with a trained Deep Neural Network policy.
    Evaluates state vector and samples argmax_a P(a | s).
    """

    def __init__(self, model: PolicyNetwork):
        self.model = model
        self.model.eval()

    def plan_next_step(self, goal: Dict[str, str], memory: AgentMemory) -> PlannedAction:
        target_item = goal.get("item", "box")
        target_room = goal.get("target_room", "Room B")

        # 1. Goal satisfaction check
        if memory.known_item_locations.get(target_item) == target_room and target_item not in memory.inventory:
            return PlannedAction(
                thought=f"Neural Policy detected terminal condition: '{target_item}' is in '{target_room}'.",
                tool_name="look",
                arguments={},
                is_terminal=True
            )

        # 2. Vectorize internal belief state
        state_tensor = StateEncoder.encode(memory, goal).unsqueeze(0)  # shape: [1, 10]

        # 3. Neural inference
        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            action_idx = int(torch.argmax(probs).item())
            confidence = float(probs[action_idx].item())

        action_info = ActionCatalog.get_action_details(action_idx)

        thought_str = (
            f"Neural Policy output action {action_idx} ('{action_info['desc']}') "
            f"with confidence {confidence * 100:.1f}%."
        )

        return PlannedAction(
            thought=thought_str,
            tool_name=action_info["tool"],
            arguments=dict(action_info["args"]),
            is_terminal=False
        )