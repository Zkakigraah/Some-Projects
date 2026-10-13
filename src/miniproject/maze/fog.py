import time
import os
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass

from miniproject.maze.grid import Maze, Coordinate, Action, CellType
from miniproject.maze.visualizer import TerminalColors


class FogCellState:
    UNKNOWN = "?"   # Sương mù chưa khám phá
    EMPTY = " "     # Đường đi đã xác nhận
    WALL = "#"      # Tường chắn đã phát hiện
    START = "A"
    GOAL = "G"


@dataclass
class Observation:
    """Quan sát cảm biến cục bộ trả về cho Agent tại bước t."""
    agent_pos: Coordinate
    goal_pos: Optional[Coordinate]
    local_window: List[List[str]]  # Cửa sổ (2R+1) x (2R+1) quanh Agent
    radius: int
    newly_revealed_cells: List[Tuple[Coordinate, str]]


class FogOfWarEnvironment:
    """
    Môi trường POMDP (Partially Observable Markov Decision Process):
    - Che giấu toàn bộ bản đồ gốc sau bức màn sương mù Fog of War ('?').
    - Agent chỉ có cảm biến cục bộ trong bán kính fov_radius (mặc định R=1 -> cửa sổ 3x3).
    - Tự động duy trì Bản đồ nhận thức nội tâm (Internal Belief Map) khi Agent di chuyển.
    """

    def __init__(
        self,
        maze: Maze,
        fov_radius: int = 1,
        known_goal: bool = True
    ):
        self.maze = maze
        self.height = maze.height
        self.width = maze.width
        self.fov_radius = fov_radius
        self.known_goal = known_goal

        assert maze.start is not None and maze.goal is not None
        self.agent_pos: Coordinate = maze.start
        self.goal_pos: Coordinate = maze.goal

        # Bản đồ nội tâm của Agent (khởi tạo toàn bộ là sương mù '?')
        self.internal_map: List[List[str]] = [
            [FogCellState.UNKNOWN for _ in range(self.width)]
            for _ in range(self.height)
        ]

        # Đánh dấu các ô đã từng đặt chân tới
        self.visited_history: List[Coordinate] = [self.agent_pos]
        self.explored_cells: Set[Coordinate] = set()

        if self.known_goal:
            self.internal_map[self.goal_pos[0]][self.goal_pos[1]] = FogCellState.GOAL

        # Cập nhật tầm nhìn tại điểm xuất phát ban đầu
        self._update_fov()

    def reset(self) -> Observation:
        assert self.maze.start is not None
        self.agent_pos = self.maze.start
        self.internal_map = [
            [FogCellState.UNKNOWN for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.visited_history = [self.agent_pos]
        self.explored_cells = set()
        if self.known_goal:
            self.internal_map[self.goal_pos[0]][self.goal_pos[1]] = FogCellState.GOAL
        return self._update_fov()

    def _update_fov(self) -> Observation:
        """Cảm biến quét và làm tan sương mù trong bán kính R."""
        ar, ac = self.agent_pos
        R = self.fov_radius
        newly_revealed = []
        local_window = []

        for dr in range(-R, R + 1):
            row_chars = []
            for dc in range(-R, R + 1):
                r, c = ar + dr, ac + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    real_char = self.maze.grid[r][c]
                    # Nếu ô này đang là sương mù, nay được giải mã
                    if self.internal_map[r][c] == FogCellState.UNKNOWN:
                        self.internal_map[r][c] = real_char
                        newly_revealed.append(((r, c), real_char))
                    self.explored_cells.add((r, c))
                    row_chars.append(real_char)
                else:
                    row_chars.append(FogCellState.WALL)
            local_window.append(row_chars)

        return Observation(
            agent_pos=self.agent_pos,
            goal_pos=self.goal_pos if self.known_goal else None,
            local_window=local_window,
            radius=R,
            newly_revealed_cells=newly_revealed,
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Thực hiện một bước di chuyển:
        - Trả về: (Quan sát mới, Phần thưởng, Kết thúc?, Thông tin phụ).
        """
        dr, dc = action.delta
        next_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        # Kiểm tra va chạm với tường thực tế
        if not self.maze.is_traversable(next_pos):
            # Cập nhật ô tường này vào bản đồ nội tâm
            if self.maze.is_in_bounds(next_pos):
                self.internal_map[next_pos[0]][next_pos[1]] = FogCellState.WALL
            obs = self._update_fov()
            return obs, -5.0, False, {"collision": True, "reason": "BUMPED_WALL"}

        self.agent_pos = next_pos
        self.visited_history.append(self.agent_pos)
        obs = self._update_fov()

        is_done = (self.agent_pos == self.goal_pos)
        reward = 100.0 if is_done else -1.0

        return obs, reward, is_done, {"collision": False, "reason": "MOVED"}

    def get_discovered_coverage(self) -> float:
        """Tỷ lệ % diện tích mê cung đã được làm tan sương mù."""
        total_cells = self.height * self.width
        revealed = sum(
            1 for r in range(self.height) for c in range(self.width)
            if self.internal_map[r][c] != FogCellState.UNKNOWN
        )
        return (revealed / total_cells) * 100.0

    def render_ansi(self, show_trail: bool = True) -> str:
        """Hiển thị giao diện màn sương mù thời gian thực với màu sắc ANSI."""
        trail_set = set(self.visited_history)
        lines = []

        for r in range(self.height):
            row_str = []
            for c in range(self.width):
                pos = (r, c)
                cell = self.internal_map[r][c]

                if pos == self.agent_pos:
                    row_str.append(f"{TerminalColors.YELLOW}{TerminalColors.BOLD}@{TerminalColors.RESET}")
                elif pos == self.goal_pos and (self.known_goal or cell == FogCellState.GOAL):
                    row_str.append(f"{TerminalColors.RED}{TerminalColors.BOLD}G{TerminalColors.RESET}")
                elif pos == self.maze.start:
                    row_str.append(f"{TerminalColors.GREEN}{TerminalColors.BOLD}A{TerminalColors.RESET}")
                elif cell == FogCellState.UNKNOWN:
                    row_str.append(f"{TerminalColors.DIM}░{TerminalColors.RESET}")  # Sương mù
                elif cell == FogCellState.WALL:
                    row_str.append(f"{TerminalColors.WALL_STYLE}#{TerminalColors.RESET}")
                elif show_trail and pos in trail_set:
                    row_str.append(f"{TerminalColors.CYAN}·{TerminalColors.RESET}")
                else:
                    row_str.append(" ")
            lines.append("".join(row_str))

        return "\n".join(lines)