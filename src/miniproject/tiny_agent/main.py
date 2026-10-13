import sys
import os

# Ensure project root (containing 'miniproject') is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from miniproject.tiny_agent.environment import MultiRoomEnvironment
from miniproject.tiny_agent.tools.base import ToolRegistry
from miniproject.tiny_agent.tools.actions import LookTool, MoveTool, PickUpTool, DropTool, OpenTool
from miniproject.tiny_agent.memory import AgentMemory
from miniproject.tiny_agent.planner import GoalDirectedPlanner, LLMPlanner
from miniproject.tiny_agent.agent import TinyAgent
from miniproject.tiny_agent.neural_policy import PolicyNetwork, PolicyTrainer, NeuralPolicyPlanner


def build_agent(planner, env=None, name="Agent") -> TinyAgent:
    """Helper factory to assemble an agent with standard environment and tools."""
    env = env or MultiRoomEnvironment()
    registry = ToolRegistry()
    registry.register(LookTool(env))
    registry.register(MoveTool(env))
    registry.register(PickUpTool(env))
    registry.register(DropTool(env))
    registry.register(OpenTool(env))

    memory = AgentMemory()
    return TinyAgent(
        name=name,
        env=env,
        tools=registry,
        planner=planner,
        memory=memory,
        max_steps=12
    )


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 TINY AGENT: THE 3 PARADIGMS                   ║
    ║   1. Symbolic Planner  ──►  2. Neural MLP  ──►  3. LLM ReAct  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    goal = {"item": "box", "target_room": "Room B"}

    # ==============================================================
    # PARADIGM 1: SYMBOLIC (RULE-BASED STATE MACHINE)
    # ==============================================================
    print("🔹 [1/3] PARADIGM 1: Symbolic / Goal-Directed Planner")
    agent_symbolic = build_agent(planner=GoalDirectedPlanner(), name="Symbolic-Agent")
    success_sym = agent_symbolic.run(goal)

    # ==============================================================
    # PARADIGM 2: NEURAL POLICY (MLP via Imitation Learning)
    # ==============================================================
    print("\n" + "="*65)
    print("🧠 [2/3] PARADIGM 2: Neural Policy (PyTorch MLP)")
    print("="*65)
    policy_net = PolicyNetwork(state_dim=10, hidden_dim=64, num_actions=7)
    PolicyTrainer.train(policy_net, epochs=180, lr=0.005)
    agent_neural = build_agent(planner=NeuralPolicyPlanner(policy_net), name="Neural-Agent")
    success_neural = agent_neural.run(goal)

    # ==============================================================
    # PARADIGM 3: LLM REASONER (ReAct: Reason + Act)
    # ==============================================================
    print("\n" + "="*65)
    print("💬 [3/3] PARADIGM 3: LLM Planner (In-Context ReAct Chain)")
    print("="*65)
    # Pass api_key="sk-..." and base_url if you have a live endpoint, or use default simulator
    llm_planner = LLMPlanner(model_name="gpt-4o-react")
    agent_llm = build_agent(planner=llm_planner, name="LLM-ReAct-Agent")
    success_llm = agent_llm.run(goal)

    # ==============================================================
    # CROSS-PARADIGM COMPARISON TABLE
    # ==============================================================
    print("\n" + "#"*65)
    print("📊 ARCHITECTURAL COMPARISON: 3 SIDES OF COGNITIVE PLANNING")
    print("#"*65)
    print(f"| {'Feature':<22} | {'Symbolic Planner':<16} | {'Neural Policy':<14} | {'LLM ReAct':<14} |")
    print(f"|{'-'*24}|{'-'*18}|{'-'*16}|{'-'*16}|")
    print(f"| {'Steps to Goal':<22} | {agent_symbolic.step_counter:<16} | {agent_neural.step_counter:<14} | {agent_llm.step_counter:<14} |")
    print(f"| {'Reasoning Core':<22} | {'Explicit Rules':<16} | {'Learned Weights':<14} | {'In-Context ReAct':<14} |")
    print(f"| {'Interpretability':<22} | {'High (Code)':<16} | {'Low (Black-box)':<14} | {'Highest (Natural)':<14} |")
    print(f"| {'Inference Cost':<22} | {'Microseconds':<16} | {'Sub-millisecond':<14} | {'High (Tokens/s)':<14} |")
    print(f"| {'Out-of-Domain':<22} | {'Fails / Brittle':<16} | {'Unstable':<14} | {'Generalizes':<14} |")
    print("#"*65)


if __name__ == "__main__":
    main()