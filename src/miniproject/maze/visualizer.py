import time
import os
from typing import List, Set, Tuple
from miniproject.maze.grid import Maze, Coordinate, CellType
from miniproject.maze.solvers import SearchResult


class TerminalColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    WALL_STYLE = "\033[90m"
    EXPLORED_STYLE = "\033[36m"
    PATH_STYLE = "\033[93m\033[1m"


class MazeVisualizer:
    """Công cụ render trực quan hóa mê cung, tiến trình duyệt và đường đi tối ưu."""

    @staticmethod
    def render_solved_maze(
        maze: Maze,
        result: SearchResult,
        show_explored: bool = True
    ) -> str:
        """Tạo chuỗi ký tự hiển thị toàn bộ maze với màu ANSI."""
        path_set: Set[Coordinate] = set(result.path)
        explored_set: Set[Coordinate] = set(result.explored_order)

        output_lines = []
        for r in range(maze.height):
            line_str = []
            for c in range(maze.width):
                pos = (r, c)
                char = maze.grid[r][c]

                if pos == maze.start:
                    line_str.append(f"{TerminalColors.GREEN}{TerminalColors.BOLD}A{TerminalColors.RESET}")
                elif pos == maze.goal:
                    line_str.append(f"{TerminalColors.RED}{TerminalColors.BOLD}G{TerminalColors.RESET}")
                elif pos in path_set:
                    line_str.append(f"{TerminalColors.PATH_STYLE}*{TerminalColors.RESET}")
                elif show_explored and pos in explored_set:
                    line_str.append(f"{TerminalColors.EXPLORED_STYLE}·{TerminalColors.RESET}")
                elif char == CellType.WALL:
                    line_str.append(f"{TerminalColors.WALL_STYLE}#{TerminalColors.RESET}")
                elif char == CellType.MUD:
                    line_str.append(f"{TerminalColors.MAGENTA}~{TerminalColors.RESET}")
                else:
                    line_str.append(" ")
            output_lines.append("".join(line_str))

        return "\n".join(output_lines)

    @staticmethod
    def print_comparison_table(results: List[Tuple[str, SearchResult]]) -> None:
        """In bảng so sánh số liệu thực nghiệm giữa các thuật toán."""
        header = f"{'Thuật toán':<32} | {'Độ dài Path':<12} | {'Tổng Cost':<10} | {'Nodes duyệt':<12} | {'Thời gian (ms)':<15} | {'Tối ưu?'}"
        separator = "-" * len(header)

        print("\n" + separator)
        print(f"{TerminalColors.BOLD}{header}{TerminalColors.RESET}")
        print(separator)

        successful = [res for _, res in results if res.success]
        min_cost = min(res.total_cost for res in successful) if successful else 0.0

        for name, res in results:
            if not res.success:
                print(f"{name:<32} | {'THẤT BẠI':<12} | {'-':<10} | {res.nodes_explored_count:<12} | {res.runtime_ms:<15.4f} | Không")
                continue

            path_len = len(res.path)
            is_optimal = "Tối ưu" if abs(res.total_cost - min_cost) < 1e-6 else "Không tối ưu"
            opt_color = TerminalColors.GREEN if is_optimal == "Tối ưu" else TerminalColors.YELLOW

            print(
                f"{name:<32} | "
                f"{path_len:<12} | "
                f"{res.total_cost:<10.1f} | "
                f"{res.nodes_explored_count:<12} | "
                f"{res.runtime_ms:<15.4f} | "
                f"{opt_color}{is_optimal}{TerminalColors.RESET}"
            )
        print(separator + "\n")