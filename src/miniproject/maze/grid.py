import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Set, Dict

Coordinate = Tuple[int, int]


@dataclass
class CharMapping:
    """
    Cấu hình bộ ký tự tùy biến cho Maze.
    Hỗ trợ đầy đủ ký tự ASCII và toàn bộ Unicode Box-Drawing.
    """
    walls: Set[str] = field(default_factory=lambda: {
        "#", "1", "X", "x", "█", "+", "|", "-",
        "┌", "─", "┬", "┐", "│", "└", "┴", "┘", "┤", "├", "┼",
        "╴", "╵", "╶", "╷", "╭", "╮", "╯", "╰", "═", "║"
    })
    starts: Set[str] = field(default_factory=lambda: {"A", "S", "s", "2"})
    goals: Set[str] = field(default_factory=lambda: {"G", "E", "e", "3", "F", "f"})
    muds: Set[str] = field(default_factory=lambda: {"~", "m", "M"})
    empties: Set[str] = field(default_factory=lambda: {" ", "0", ".", "_"})


class Action(Enum):
    """4 hướng di chuyển cơ bản của agent trong Grid World 2D."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def delta(self) -> Tuple[int, int]:
        return self.value


class CellType:
    WALL = "#"
    EMPTY = " "
    START = "A"
    GOAL = "G"
    MUD = "~"


class Maze:
    """
    Môi trường Maze biểu diễn dưới dạng 2D Grid.
    Tự động chuẩn hóa dòng, loại bỏ các cột khoảng trắng đệm ngoài rìa,
    và tự động nhận diện cặp Start/Goal tối ưu.
    """

    def __init__(
        self,
        raw_layout: str,
        mud_cost: float = 3.0,
        mapping: Optional[CharMapping] = None,
        custom_start: Optional[Coordinate] = None,
        custom_goal: Optional[Coordinate] = None,
    ):
        raw_lines = raw_layout.strip("\r\n").splitlines()
        self.raw_layout = [line for line in raw_lines if line.strip()]

        self.height = len(self.raw_layout)
        self.width = max(len(row) for row in self.raw_layout) if self.height > 0 else 0
        self.mud_cost = mud_cost
        self.mapping = mapping or CharMapping()

        self.grid: List[List[str]] = []
        self.start: Optional[Coordinate] = custom_start
        self.goal: Optional[Coordinate] = custom_goal

        self._parse_layout()
        self._trim_empty_boundary_columns()

        if self.start is None or self.goal is None:
            self._auto_detect_endpoints()

        if self.start is None:
            raise ValueError("Không thể xác định điểm bắt đầu! Vui lòng chỉ định qua CLI (--start row,col).")
        if self.goal is None:
            raise ValueError("Không thể xác định điểm kết thúc! Vui lòng chỉ định qua CLI (--goal row,col).")

    @classmethod
    def from_file(
        cls,
        filepath: str,
        mud_cost: float = 3.0,
        mapping: Optional[CharMapping] = None,
        custom_start: Optional[Coordinate] = None,
        custom_goal: Optional[Coordinate] = None,
    ) -> "Maze":
        """Khởi tạo Maze trực tiếp từ một đường dẫn file text."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls(
            raw_layout=content,
            mud_cost=mud_cost,
            mapping=mapping,
            custom_start=custom_start,
            custom_goal=custom_goal
        )

    def _trim_empty_boundary_columns(self) -> None:
        """Loại bỏ các cột ngoài cùng nếu 100% cột đó là khoảng trắng (padding thừa ngoài rìa file)."""
        if self.width <= 2:
            return

        # Kiểm tra cột phải cùng: nếu TOÀN BỘ cột là khoảng trắng đệm thì cắt bỏ
        while self.width > 2:
            last_col = self.width - 1
            is_all_empty = all(self.grid[r][last_col] == CellType.EMPTY for r in range(self.height))
            if is_all_empty:
                for r in range(self.height):
                    self.grid[r].pop()
                self.width -= 1
            else:
                break

        # Kiểm tra cột trái cùng
        while self.width > 2:
            is_first_all_empty = all(self.grid[r][0] == CellType.EMPTY for r in range(self.height))
            if is_first_all_empty:
                for r in range(self.height):
                    self.grid[r].pop(0)
                self.width -= 1
            else:
                break

    def _auto_detect_endpoints(self) -> None:
        """
        Tự động xác định Start và Goal:
        1. Tìm toàn bộ các khe hở (openings) trên viền ngoài.
        2. Chọn cặp khe hở có khoảng cách Euclid xa nhất (ví dụ: đỉnh trên và đáy dưới).
        3. Nếu viền kín, chọn ô trống góc trên-trái và góc dưới-phải.
        """
        perimeter_openings: List[Coordinate] = []

        # Kiểm tra 4 cạnh viền ngoài
        for c in range(self.width):
            if self.grid[0][c] == CellType.EMPTY:
                perimeter_openings.append((0, c))
            if self.grid[self.height - 1][c] == CellType.EMPTY:
                perimeter_openings.append((self.height - 1, c))

        for r in range(1, self.height - 1):
            if self.grid[r][0] == CellType.EMPTY:
                perimeter_openings.append((r, 0))
            if self.grid[r][self.width - 1] == CellType.EMPTY:
                perimeter_openings.append((r, self.width - 1))

        if len(perimeter_openings) >= 2:
            best_pair = (perimeter_openings[0], perimeter_openings[-1])
            max_dist_sq = -1.0
            for i in range(len(perimeter_openings)):
                for j in range(i + 1, len(perimeter_openings)):
                    p1 = perimeter_openings[i]
                    p2 = perimeter_openings[j]
                    d_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                    if d_sq > max_dist_sq:
                        max_dist_sq = d_sq
                        best_pair = (p1, p2)

            if self.start is None:
                self.start = best_pair[0]
                self.grid[self.start[0]][self.start[1]] = CellType.START
            if self.goal is None:
                self.goal = best_pair[1]
                self.grid[self.goal[0]][self.goal[1]] = CellType.GOAL
            return

        # Trường hợp viền ngoài hoàn toàn kín (như maze_2.txt)
        empty_cells: List[Coordinate] = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == CellType.EMPTY:
                    empty_cells.append((r, c))

        if empty_cells:
            empty_cells.sort(key=lambda pos: (pos[0] + pos[1], pos[0]))
            if self.start is None:
                self.start = empty_cells[0]
                self.grid[self.start[0]][self.start[1]] = CellType.START
            if self.goal is None:
                self.goal = empty_cells[-1]
                self.grid[self.goal[0]][self.goal[1]] = CellType.GOAL

    def _parse_layout(self) -> None:
        """Phân tích ký tự và chuẩn hóa thành ma trận CellType."""
        for r, line in enumerate(self.raw_layout):
            row_cells: List[str] = []
            for c in range(self.width):
                char = line[c] if c < len(line) else CellType.WALL
                if char in self.mapping.starts:
                    self.start = (r, c)
                    normalized = CellType.START
                elif char in self.mapping.goals:
                    self.goal = (r, c)
                    normalized = CellType.GOAL
                elif char in self.mapping.walls:
                    normalized = CellType.WALL
                elif char in self.mapping.muds:
                    normalized = CellType.MUD
                else:
                    normalized = CellType.EMPTY
                row_cells.append(normalized)
            self.grid.append(row_cells)

    def is_in_bounds(self, pos: Coordinate) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_traversable(self, pos: Coordinate) -> bool:
        if not self.is_in_bounds(pos):
            return False
        r, c = pos
        return self.grid[r][c] != CellType.WALL

    def get_step_cost(self, pos: Coordinate) -> float:
        r, c = pos
        cell = self.grid[r][c]
        if cell == CellType.MUD:
            return self.mud_cost
        return 1.0

    def get_neighbors(self, pos: Coordinate) -> List[Tuple[Action, Coordinate, float]]:
        neighbors = []
        r, c = pos
        for action in Action:
            dr, dc = action.delta
            next_pos = (r + dr, c + dc)
            if self.is_traversable(next_pos):
                cost = self.get_step_cost(next_pos)
                neighbors.append((action, next_pos, cost))
        return neighbors

    def __repr__(self) -> str:
        return f"Maze(size={self.height}x{self.width}, start={self.start}, goal={self.goal})"