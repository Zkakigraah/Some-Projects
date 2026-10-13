import random
from typing import List, Tuple, Optional
from miniproject.maze.grid import Maze, CellType, Coordinate


class MazeGenerator:
    """
    Bộ sinh ngẫu nhiên mê cung hợp lệ (luôn có đường đi từ Start tới Goal).
    Sử dụng thuật toán Randomized Depth-First Search (Recursive Backtracker).
    """

    @staticmethod
    def generate(
        height: int = 11,
        width: int = 11,
        loop_factor: float = 0.05,
        seed: Optional[int] = None
    ) -> Maze:
        """
        Sinh ngẫu nhiên một maze kích thước height x width (khuyến nghị số lẻ).
        - loop_factor: Tỷ lệ phá bỏ thêm tường để tạo vòng lặp (giảm tính chất cây đơn đạo).
        """
        if seed is not None:
            random.seed(seed)

        # Đảm bảo kích thước là số lẻ để tường bao và đường đi xen kẽ chuẩn xác
        h = height if height % 2 == 1 else height + 1
        w = width if width % 2 == 1 else width + 1

        # Khởi tạo toàn bộ là tường bao
        grid: List[List[str]] = [[CellType.WALL for _ in range(w)] for _ in range(h)]

        # DFS Backtracker để đục đường đi
        start_r, start_c = 1, 1
        grid[start_r][start_c] = CellType.EMPTY
        stack: List[Coordinate] = [(start_r, start_c)]
        visited = {(start_r, start_c)}

        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            cr, cc = stack[-1]
            neighbors = []

            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 1 <= nr < h - 1 and 1 <= nc < w - 1 and (nr, nc) not in visited:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = random.choice(neighbors)
                # Đục tường ở giữa ô hiện tại và ô kế tiếp
                wall_r, wall_c = cr + dr // 2, cc + dc // 2
                grid[wall_r][wall_c] = CellType.EMPTY
                grid[nr][nc] = CellType.EMPTY

                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Thêm loop bằng cách ngẫu nhiên phá tường ngăn
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grid[r][c] == CellType.WALL and random.random() < loop_factor:
                    # Chỉ phá tường nếu 2 bên đối diện là đường trống
                    horizontal_open = (grid[r][c - 1] == CellType.EMPTY and grid[r][c + 1] == CellType.EMPTY)
                    vertical_open = (grid[r - 1][c] == CellType.EMPTY and grid[r + 1][c] == CellType.EMPTY)
                    if horizontal_open or vertical_open:
                        grid[r][c] = CellType.EMPTY

        # Đặt điểm bắt đầu A và điểm đích G
        grid[1][1] = CellType.START
        grid[h - 2][w - 2] = CellType.GOAL

        # Chuyển đổi thành chuỗi layout để khởi tạo Maze
        raw_layout = "\n".join("".join(row) for row in grid)
        return Maze(raw_layout)