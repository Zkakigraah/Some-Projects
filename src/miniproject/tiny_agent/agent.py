import time
from typing import Dict, Any, Optional
from miniproject.tiny_agent.environment import MultiRoomEnvironment, Observation
from miniproject.tiny_agent.tools.base import ToolRegistry, ToolResult
from miniproject.tiny_agent.memory import AgentMemory
from miniproject.tiny_agent.planner import BasePlanner, PlannedAction


class TinyAgent:
    """
    Autonomous Tiny Agent coordinating:
      Observation -> State/Memory -> Reasoning/Planner -> Action Execution -> Environment
    """

    def __init__(
        self,
        name: str,
        env: MultiRoomEnvironment,
        tools: ToolRegistry,
        planner: BasePlanner,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 15
    ):
        self.name = name
        self.env = env
        self.tools = tools
        self.planner = planner
        self.memory = memory or AgentMemory()
        self.max_steps = max_steps
        self.step_counter = 0

    def step(self, goal: Dict[str, str]) -> bool:
        """
        Executes one full cognitive cycle:
          1. Think (Reasoning over Goal & Memory)
          2. Act (Tool dispatch)
          3. Perceive (Observe environment outcome)
          4. Update (Belief state / Episodic memory)
        """
        self.step_counter += 1
        print(f"\n{'='*20} STEP {self.step_counter} {'='*20}")

        # 1. REASONING / PLANNING
        plan: PlannedAction = self.planner.plan_next_step(goal, self.memory)
        print(f"🧠 [Thought]: {plan.thought}")

        if plan.is_terminal:
            print("🏁 [Goal]: Agent has satisfied all conditions and declared mission complete.")
            return True

        # 2. ACTION EXECUTION
        print(f"🛠️  [Action]: Invoking tool '{plan.tool_name}' with args {plan.arguments}")
        result: ToolResult = self.tools.execute(plan.tool_name, **plan.arguments)
        print(f"📦 [Result]: {result}")

        # 3. PERCEPTION & OBSERVATION
        # Tools attach updated observation in metadata, or we probe the environment
        obs: Observation = result.metadata.get("observation") if result.metadata else self.env.get_observation()

        # 4. MEMORY / BELIEF STATE UPDATE
        self.memory.update_from_observation(
            obs=obs,
            step=self.step_counter,
            thought=plan.thought,
            tool=plan.tool_name,
            args=plan.arguments
        )
        print(f"💾 [Belief State]: {self.memory.get_state_summary()}")

        return False

    def run(self, goal: Dict[str, str]) -> bool:
        """Runs the loop until goal is achieved or max steps are exhausted."""
        print(f"🚀 Initializing Agent '{self.name}' with Goal: {goal}")
        
        # Initial perception to calibrate memory
        initial_obs = self.env.get_observation("System start.")
        self.memory.update_from_observation(
            obs=initial_obs,
            step=0,
            thought="Bootstrapping sensory input.",
            tool="init",
            args={}
        )

        completed = False
        while self.step_counter < self.max_steps:
            done = self.step(goal)
            if done or self.env.is_goal_satisfied(goal.get("item", "box"), goal.get("target_room", "Room B")):
                completed = True
                break

        print("\n" + "="*50)
        if completed:
            print(f"🎉 MISSION SUCCESS in {self.step_counter} steps!")
        else:
            print(f"❌ MISSION FAILED: Step limit reached ({self.max_steps}).")
        print("="*50)
        return completed