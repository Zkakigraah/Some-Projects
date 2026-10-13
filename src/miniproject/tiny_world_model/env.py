import copy
from typing import Dict, List, Optional, Set, Tuple


class GridWorldEnv:
    """
    GridWorld 2D đóng vai trò là "Ground Truth Environment" (Thế giới thực).
    Agent di chuyển trong lưới tọa độ (row, col) với các ô chướng ngại vật (walls)
    và mục tiêu (goal).
    """

    # 4 hành động cơ bản: UP, RIGHT, DOWN, LEFT
    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # Lên
        1: (0, 1),   # Phải
        2: (1, 0),   # Xuống
        3: (0, -1),  # Trái
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "UP",
        1: "RIGHT",
        2: "DOWN",
        3: "LEFT",
    }

    def __init__(
        self,
        height: int = 6,
        width: int = 6,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (5, 5),
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        max_steps: int = 40,
    ) -> None:
        self.height = height
        self.width = width
        self.num_states = height * width
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.max_steps = max_steps

        # Thiết lập chướng ngại vật mặc định nếu không truyền vào
        if obstacles is None:
            self.obstacles: Set[Tuple[int, int]] = {
                (1, 2), (2, 2), (3, 2),
                (3, 4), (4, 4), (1, 4)
            }
        else:
            self.obstacles = copy.deepcopy(obstacles)

        self.agent_pos = self.start_pos
        self.step_count = 0

    def pos_to_state_idx(self, pos: Tuple[int, int]) -> int:
        """Chuyển đổi tọa độ 2D (r, c) sang chỉ số 1D phẳng."""
        r, c = pos
        return r * self.width + c

    def state_idx_to_pos(self, state_idx: int) -> Tuple[int, int]:
        """Chuyển đổi chỉ số 1D phẳng về tọa độ 2D (r, c)."""
        r = state_idx // self.width
        c = state_idx % self.width
        return (r, c)

    def reset(self, random_start: bool = False) -> int:
        """Khởi động lại môi trường và trả về state index ban đầu."""
        self.step_count = 0
        if not random_start:
            self.agent_pos = self.start_pos
        else:
            # Chọn ngẫu nhiên một ô trống hợp lệ
            free_cells = [
                (r, c)
                for r in range(self.height)
                for c in range(self.width)
                if (r, c) not in self.obstacles and (r, c) != self.goal_pos
            ]
            import random
            self.agent_pos = random.choice(free_cells)

        return self.pos_to_state_idx(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Thực hiện một bước trong môi trường thực.
        Returns:
            next_state_idx (int), reward (float), done (bool), info (dict)
        """
        assert action in self.ACTIONS, f"Hành động không hợp lệ: {action}"
        self.step_count += 1

        dr, dc = self.ACTIONS[action]
        next_r = self.agent_pos[0] + dr
        next_c = self.agent_pos[1] + dc
        target_pos = (next_r, next_c)

        # Kiểm tra va chạm biên hoặc chướng ngại vật
        hit_wall = (
            next_r < 0
            or next_r >= self.height
            or next_c < 0
            or next_c >= self.width
            or target_pos in self.obstacles
        )

        if not hit_wall:
            self.agent_pos = target_pos

        # Đánh giá phần thưởng và điều kiện kết thúc
        reached_goal = (self.agent_pos == self.goal_pos)
        is_timeout = (self.step_count >= self.max_steps)
        done = reached_goal or is_timeout

        if reached_goal:
            reward = 10.0
        elif hit_wall:
            reward = -1.0  # Phạt đâm tường
        else:
            reward = -0.1  # Phạt tiêu tốn thời gian di chuyển

        info = {
            "agent_pos": self.agent_pos,
            "hit_wall": hit_wall,
            "reached_goal": reached_goal,
            "step_count": self.step_count,
        }

        return self.pos_to_state_idx(self.agent_pos), reward, done, info