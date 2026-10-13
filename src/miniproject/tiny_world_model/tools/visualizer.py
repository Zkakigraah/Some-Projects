from typing import Dict, List, Optional, Set, Tuple
from miniproject.tiny_world_model.env import GridWorldEnv


class TerminalVisualizer:
    """
    Công cụ kết xuất đồ họa ANSI Terminal, hiển thị:
    - Bản đồ GridWorld thực tế
    - Nhánh tương lai giả lập (Imagined Rollouts) của World Model
    """

    COLOR_RESET = "\033[0m"
    COLOR_AGENT = "\033[92m"      # Xanh lá
    COLOR_GOAL = "\033[93m"       # Vàng
    COLOR_WALL = "\033[90m"       # Xám tối
    COLOR_TRAJECTORY = "\033[96m" # Cyan sáng

    @classmethod
    def render_grid(
        cls,
        env: GridWorldEnv,
        agent_pos: Optional[Tuple[int, int]] = None,
        path: Optional[List[Tuple[int, int]]] = None,
        title: str = "GridWorld",
    ) -> str:
        agent_p = agent_pos if agent_pos is not None else env.agent_pos
        path_set: Set[Tuple[int, int]] = set(path) if path else set()

        lines = [f"=== {title} ==="]
        for r in range(env.height):
            row_chars = []
            for c in range(env.width):
                pos = (r, c)
                if pos == agent_p:
                    row_chars.append(f"{cls.COLOR_AGENT}[A]{cls.COLOR_RESET}")
                elif pos == env.goal_pos:
                    row_chars.append(f"{cls.COLOR_GOAL}[G]{cls.COLOR_RESET}")
                elif pos in env.obstacles:
                    row_chars.append(f"{cls.COLOR_WALL}[#]{cls.COLOR_RESET}")
                elif pos in path_set:
                    row_chars.append(f"{cls.COLOR_TRAJECTORY}[*]{cls.COLOR_RESET}")
                else:
                    row_chars.append(" . ")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

    @classmethod
    def display_planning_branches(
        cls,
        env: GridWorldEnv,
        current_state: int,
        branch_analysis: Dict[int, Dict[str, object]],
        chosen_action: int,
    ) -> None:
        curr_pos = env.state_idx_to_pos(current_state)
        print(f"\n[AGENT THOUGHT PROCESS] Trạng thái hiện tại: {curr_pos}")
        print("Mô hình đang mô phỏng các tương lai khả dĩ trong trí tưởng tượng:")

        for action_idx, info in branch_analysis.items():
            action_name = env.ACTION_NAMES[action_idx]
            ret = info["expected_return"]
            path_indices = info["best_imagined_path"]
            path_coords = [env.state_idx_to_pos(idx) for idx in path_indices]

            marker = "==> [CHOSEN]" if action_idx == chosen_action else "   "
            coords_str = " -> ".join([str(p) for p in path_coords[:5]])
            print(
                f"{marker} Nhánh {action_name:<5} | "
                f"Predicted Return: {ret:6.2f} | Path: {coords_str}..."
            )

    @classmethod
    def render_value_heatmap(cls, env: GridWorldEnv, actor_critic: object) -> str:
        """
        Trực quan hóa hàm giá trị Critic V(s) tự học qua TD-Learning trên toàn bộ lưới.
        Chứng minh dốc thế năng hình thành tự nhiên hướng về mục tiêu mà KHÔNG cần hàm Heuristic.
        """
        import torch

        actor_critic.eval()
        device = next(actor_critic.parameters()).device

        lines = ["=== MA TRẬN GIÁ TRỊ TỰ HỌC (LEARNED VALUE FUNCTION V(s)) ==="]
        for r in range(env.height):
            row_cells = []
            for c in range(env.width):
                pos = (r, c)
                if pos in env.obstacles:
                    row_cells.append(f"{cls.COLOR_WALL}  ####  {cls.COLOR_RESET}")
                elif pos == env.goal_pos:
                    row_cells.append(f"{cls.COLOR_GOAL} [GOAL] {cls.COLOR_RESET}")
                else:
                    s_idx = env.pos_to_state_idx(pos)
                    with torch.no_grad():
                        s_t = torch.tensor([s_idx], dtype=torch.long, device=device)
                        _, val = actor_critic(s_t)
                        val_num = val.item()
                    row_cells.append(f"{val_num:7.2f} ")
            lines.append(" | ".join(row_cells))
        return "\n".join(lines)

    @classmethod
    def render_policy_field(cls, env: GridWorldEnv, actor_critic: object) -> str:
        """
        Hiển thị trường vector chính sách (Policy Vector Field):
        Mũi tên chỉ hướng đi tối ưu của Policy pi(s) tại mỗi ô trên bản đồ.
        """
        action_arrows = {
            0: "  ↑  ",  # UP
            1: "  →  ",  # RIGHT
            2: "  ↓  ",  # DOWN
            3: "  ←  ",  # LEFT
        }

        lines = ["=== TRƯỜNG VECTOR CHÍNH SÁCH TRONG MƠ (LEARNED POLICY FIELD pi(s)) ==="]
        for r in range(env.height):
            row_cells = []
            for c in range(env.width):
                pos = (r, c)
                if pos in env.obstacles:
                    row_cells.append(f"{cls.COLOR_WALL}[###]{cls.COLOR_RESET}")
                elif pos == env.goal_pos:
                    row_cells.append(f"{cls.COLOR_GOAL}[ G ]{cls.COLOR_RESET}")
                else:
                    s_idx = env.pos_to_state_idx(pos)
                    best_a = actor_critic.get_action(s_idx, deterministic=True)
                    row_cells.append(f"{cls.COLOR_AGENT}{action_arrows[best_a]}{cls.COLOR_RESET}")
            lines.append(" ".join(row_cells))
        return "\n".join(lines)