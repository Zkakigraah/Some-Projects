import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any
from pathlib import Path 

from miniproject.maze.grid import Maze, Coordinate, Action

DELTA_TO_ACTION_IDX = {
    (-1, 0): 0,  # UP
    (1, 0): 1,   # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,   # RIGHT
}

@dataclass
class SearchResult:
    """
    Chứa kết quả giải maze:
    - path: Danh sách các tọa độ từ Start -> Goal.
    - explored_order: Thứ tự các node được thuật toán duyệt.
    - total_cost: Tổng chi phí đường đi.
    - nodes_explored_count: Tổng số trạng thái đã được mở rộng.
    - runtime_ms: Thời gian chạy thuật toán (mili-giây).
    - success: Có tìm thấy đường đi tới Goal hay không.
    """
    path: List[Coordinate] 
    explored_order: List[Coordinate]
    total_cost: float
    nodes_explored_count: int
    runtime_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSolver(ABC):
    """Lớp trừu tượng cho mọi thuật toán tìm kiếm đường đi (Pathfinding)."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(self, maze: Maze) -> SearchResult:
        pass

    @staticmethod
    def reconstruct_path(
        came_from: Dict[Coordinate, Optional[Coordinate]],
        current: Coordinate
    ) -> List[Coordinate]:
        """Tái dựng lại đường đi từ goal ngược về start nhờ parent map."""
        path = []
        curr: Optional[Coordinate] = current
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
        return path


class BFSSolver(BaseSolver):
    def __init__(self):
        super().__init__("Breadth-First Search (BFS)")

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()
        start, goal = maze.start, maze.goal
        assert start is not None and goal is not None

        queue: deque[Coordinate] = deque([start])
        visited: Set[Coordinate] = {start}
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        explored_order: List[Coordinate] = []

        found = False
        while queue:
            current = queue.popleft()
            explored_order.append(current)
            if current == goal:
                found = True
                break

            for _, next_pos, _ in maze.get_neighbors(current):
                if next_pos not in visited:
                    visited.add(next_pos)
                    came_from[next_pos] = current
                    queue.append(next_pos)

        path = self.reconstruct_path(came_from, goal) if found else []
        total_cost = sum(maze.get_step_cost(pos) for pos in path[1:]) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
        )


class DFSSolver(BaseSolver):
    def __init__(self):
        super().__init__("Depth-First Search (DFS)")

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()
        start, goal = maze.start, maze.goal
        assert start is not None and goal is not None

        stack: List[Coordinate] = [start]
        visited: Set[Coordinate] = set()
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        explored_order: List[Coordinate] = []

        found = False
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            explored_order.append(current)

            if current == goal:
                found = True
                break

            for _, next_pos, _ in reversed(maze.get_neighbors(current)):
                if next_pos not in visited:
                    came_from[next_pos] = current
                    stack.append(next_pos)

        path = self.reconstruct_path(came_from, goal) if found else []
        total_cost = sum(maze.get_step_cost(pos) for pos in path[1:]) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
        )


class DijkstraSolver(BaseSolver):
    def __init__(self):
        super().__init__("Dijkstra (Uniform Cost Search)")

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()
        start, goal = maze.start, maze.goal
        assert start is not None and goal is not None

        counter = 0
        open_set: List[Tuple[float, int, Coordinate]] = [(0.0, counter, start)]
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        cost_so_far: Dict[Coordinate, float] = {start: 0.0}
        explored_order: List[Coordinate] = []
        closed_set: Set[Coordinate] = set()

        found = False
        while open_set:
            current_cost, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            explored_order.append(current)

            if current == goal:
                found = True
                break

            for _, next_pos, step_cost in maze.get_neighbors(current):
                new_cost = current_cost + step_cost
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    came_from[next_pos] = current
                    counter += 1
                    heapq.heappush(open_set, (new_cost, counter, next_pos))

        path = self.reconstruct_path(came_from, goal) if found else []
        total_cost = cost_so_far.get(goal, float("inf")) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
        )


class AStarSolver(BaseSolver):
    def __init__(self):
        super().__init__("A* Search (Manhattan)")

    @staticmethod
    def heuristic(a: Coordinate, b: Coordinate) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()
        start, goal = maze.start, maze.goal
        assert start is not None and goal is not None

        counter = 0
        h_start = self.heuristic(start, goal)
        open_set: List[Tuple[float, float, int, Coordinate]] = [(h_start, 0.0, counter, start)]
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        g_cost: Dict[Coordinate, float] = {start: 0.0}
        closed_set: Set[Coordinate] = set()
        explored_order: List[Coordinate] = []

        found = False
        while open_set:
            f_score, current_g, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            explored_order.append(current)

            if current == goal:
                found = True
                break

            for _, next_pos, step_cost in maze.get_neighbors(current):
                tentative_g = current_g + step_cost
                if next_pos not in g_cost or tentative_g < g_cost[next_pos]:
                    g_cost[next_pos] = tentative_g
                    came_from[next_pos] = current
                    counter += 1
                    f = tentative_g + self.heuristic(next_pos, goal)
                    heapq.heappush(open_set, (f, tentative_g, counter, next_pos))

        path = self.reconstruct_path(came_from, goal) if found else []
        total_cost = g_cost.get(goal, float("inf")) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
        )


class BidirectionalAStarSolver(BaseSolver):
    """
    Bidirectional A* Search:
    - Tìm kiếm đồng thời từ 2 đầu: Start -> và <- Goal.
    - Khi hai nhánh tìm kiếm chạm nhau (Meeting Node), ghép nối đường đi.
    - Ghi nhận độc lập 2 luồng forward và backward để phục vụ trực quan hóa 2 đầu hội tụ.
    """

    def __init__(self):
        super().__init__("Bidirectional A*")

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()

        start = maze.start
        goal = maze.goal
        assert start is not None and goal is not None

        if start == goal:
            return SearchResult([start], [start], 0.0, 1, 0.01, True)

        counter = 0
        open_fwd: List[Tuple[float, float, int, Coordinate]] = [(AStarSolver.heuristic(start, goal), 0.0, counter, start)]
        open_bwd: List[Tuple[float, float, int, Coordinate]] = [(AStarSolver.heuristic(goal, start), 0.0, counter, goal)]

        came_from_fwd: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        came_from_bwd: Dict[Coordinate, Optional[Coordinate]] = {goal: None}

        g_fwd: Dict[Coordinate, float] = {start: 0.0}
        g_bwd: Dict[Coordinate, float] = {goal: 0.0}

        closed_fwd: Set[Coordinate] = set()
        closed_bwd: Set[Coordinate] = set()

        explored_order: List[Coordinate] = []
        fwd_explored: List[Coordinate] = []
        bwd_explored: List[Coordinate] = []

        meeting_node: Optional[Coordinate] = None

        while open_fwd and open_bwd:
            # 1. Bước tiến từ phía START
            _, curr_g_fwd, _, curr_fwd = heapq.heappop(open_fwd)
            if curr_fwd not in closed_fwd:
                closed_fwd.add(curr_fwd)
                explored_order.append(curr_fwd)
                fwd_explored.append(curr_fwd)

                if curr_fwd in closed_bwd:
                    meeting_node = curr_fwd
                    break

                for _, next_pos, step_cost in maze.get_neighbors(curr_fwd):
                    new_g = curr_g_fwd + step_cost
                    if next_pos not in g_fwd or new_g < g_fwd[next_pos]:
                        g_fwd[next_pos] = new_g
                        came_from_fwd[next_pos] = curr_fwd
                        counter += 1
                        f = new_g + AStarSolver.heuristic(next_pos, goal)
                        heapq.heappush(open_fwd, (f, new_g, counter, next_pos))

            # 2. Bước tiến từ phía GOAL (Chạy đồng thời song song)
            _, curr_g_bwd, _, curr_bwd = heapq.heappop(open_bwd)
            if curr_bwd not in closed_bwd:
                closed_bwd.add(curr_bwd)
                explored_order.append(curr_bwd)
                bwd_explored.append(curr_bwd)

                if curr_bwd in closed_fwd:
                    meeting_node = curr_bwd
                    break

                for _, next_pos, step_cost in maze.get_neighbors(curr_bwd):
                    new_g = curr_g_bwd + step_cost
                    if next_pos not in g_bwd or new_g < g_bwd[next_pos]:
                        g_bwd[next_pos] = new_g
                        came_from_bwd[next_pos] = curr_bwd
                        counter += 1
                        f = new_g + AStarSolver.heuristic(next_pos, start)
                        heapq.heappush(open_bwd, (f, new_g, counter, next_pos))

        # Tái dựng đường đi nối giữa 2 đầu
        path: List[Coordinate] = []
        found = meeting_node is not None

        if found:
            curr = meeting_node
            while curr is not None:
                path.append(curr)
                curr = came_from_fwd.get(curr)
            path.reverse()

            curr = came_from_bwd.get(meeting_node)
            while curr is not None:
                path.append(curr)
                curr = came_from_bwd.get(curr)

        total_cost = sum(maze.get_step_cost(pos) for pos in path[1:]) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
            metadata={"fwd_explored": fwd_explored, "bwd_explored": bwd_explored},
        )


class JPSSolver(BaseSolver):
    """
    Jump Point Search (JPS):
    - Nhảy cóc dọc theo hành lang thẳng cho đến khi chạm góc rẽ hoặc đích đến.
    - Cắt giảm mạnh số lần thao tác với Priority Queue, cho tốc độ vượt trội trên mê cung.
    """

    def __init__(self):
        super().__init__("Jump Point Search (JPS)")

    @staticmethod
    def heuristic(a: Coordinate, b: Coordinate) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _jump(
        self,
        maze: Maze,
        current: Coordinate,
        direction: Tuple[int, int],
        goal: Coordinate
    ) -> Optional[Tuple[Coordinate, List[Coordinate], float]]:
        """Nhảy dọc theo một hướng cố định cho tới khi gặp ngã rẽ, góc ngoặt hoặc Goal."""
        dr, dc = direction
        r, c = current
        traversed: List[Coordinate] = []
        accumulated_cost = 0.0

        while True:
            r += dr
            c += dc
            pos = (r, c)

            if not maze.is_traversable(pos):
                return None

            traversed.append(pos)
            accumulated_cost += maze.get_step_cost(pos)

            if pos == goal:
                return pos, traversed, accumulated_cost

            # Kiểm tra xem ô hiện tại có phải là ngã rẽ hoặc góc ngoặt hay không
            # (tức là có thể rẽ sang hướng vuông góc)
            orthogonal_open = False
            for ortho_dr, ortho_dc in [(-dc, -dr), (dc, dr)]:
                ortho_pos = (r + ortho_dr, c + ortho_dc)
                if maze.is_traversable(ortho_pos):
                    orthogonal_open = True
                    break

            # Nếu có đường rẽ vuông góc, đây là một Jump Point!
            if orthogonal_open:
                return pos, traversed, accumulated_cost

            # Nếu phía trước là tường chắn (ngõ cụt), dừng nhảy tại đây
            if not maze.is_traversable((r + dr, c + dc)):
                return pos, traversed, accumulated_cost

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()
        start, goal = maze.start, maze.goal
        assert start is not None and goal is not None

        counter = 0
        h_start = self.heuristic(start, goal)
        open_set: List[Tuple[float, float, int, Coordinate]] = [(h_start, 0.0, counter, start)]
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        jump_segments: Dict[Tuple[Coordinate, Coordinate], List[Coordinate]] = {}
        g_cost: Dict[Coordinate, float] = {start: 0.0}
        closed_set: Set[Coordinate] = set()
        explored_order: List[Coordinate] = []

        found = False
        while open_set:
            f_score, current_g, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            explored_order.append(current)

            if current == goal:
                found = True
                break

            for action in Action:
                dr, dc = action.delta
                jump_res = self._jump(maze, current, (dr, dc), goal)
                if jump_res is not None:
                    jump_point, segment, seg_cost = jump_res
                    tentative_g = current_g + seg_cost

                    if jump_point not in g_cost or tentative_g < g_cost[jump_point]:
                        g_cost[jump_point] = tentative_g
                        came_from[jump_point] = current
                        jump_segments[(current, jump_point)] = segment
                        counter += 1
                        f = tentative_g + self.heuristic(jump_point, goal)
                        heapq.heappush(open_set, (f, tentative_g, counter, jump_point))

        # Tái dựng đường đi hoàn chỉnh từ các đoạn Jump Segment
        path: List[Coordinate] = []
        if found:
            curr_jp: Optional[Coordinate] = goal
            while curr_jp is not None and curr_jp != start:
                prev_jp = came_from.get(curr_jp)
                if prev_jp is not None:
                    seg = jump_segments.get((prev_jp, curr_jp), [curr_jp])
                    path = seg + path
                    curr_jp = prev_jp
                else:
                    break
            path = [start] + path

        total_cost = g_cost.get(goal, float("inf")) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
        )


class NeuralPolicySolver(BaseSolver):
    """
    Neural-Guided Search (AI Hybrid Solver):
    - Kết hợp mạng nơ-ron CNN và cấu trúc đồ thị mở rộng tối ưu.
    - Mạng nơ-ron đóng vai trò là 'Trực giác định hướng' (Heuristic / Prior Policy):
      gán trọng số ưu tiên cho các nhánh rẽ tiềm năng.
    - Cấu trúc Min-Heap quản lý việc mở rộng:
      Đảm bảo 100% tính hoàn chỉnh (Completeness), không bao giờ kẹt vòng lặp,
      và giải quyết các mê cung lớn hàng nghìn bước chỉ trong thời gian ngắn!
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__("Learned Policy (Neural CNN)")
        self.model_path = model_path
        self.model = None

    def _load_model_if_needed(self):
        if self.model is not None:
            return

        import torch
        from miniproject.maze.model import MazePolicyCNN

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MazePolicyCNN(in_channels=3, num_actions=4)

        target_path = Path(self.model_path) if self.model_path else (Path(__file__).resolve().parent / "maze_policy.pth")
        if target_path.exists():
            state_dict = torch.load(str(target_path), map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def solve(self, maze: Maze) -> SearchResult:
        start_time = time.perf_counter()
        self._load_model_if_needed()

        start, goal = maze.start, maze.goal
        assert start is not None and goal is not None

        if start == goal:
            return SearchResult([start], [start], 0.0, 1, 0.01, True)

        import torch

        # Priority Queue lưu: (f_score, counter, current_node)
        # f_score = g_cost + h_manhattan - alpha * neural_confidence
        counter = 0
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        open_set: List[Tuple[float, int, Coordinate]] = [(h_start, counter, start)]
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        g_cost: Dict[Coordinate, float] = {start: 0.0}
        closed_set: Set[Coordinate] = set()
        explored_order: List[Coordinate] = []

        # Cache tính toán nơ-ron để tránh suy luận trùng lặp trên cùng 1 ô
        neural_cache: Dict[Coordinate, torch.Tensor] = {}

        found = False
        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)
            explored_order.append(current)

            if current == goal:
                found = True
                break

            current_g = g_cost[current]
            neighbors = maze.get_neighbors(current)

            # Lấy logits phân phối xác suất từ Mạng Nơ-ron tại vị trí hiện tại
            # Chỉ suy luận mạng khi ở ngã rẽ có >= 2 hướng đi hợp lệ
            neural_probs = None
            if len(neighbors) >= 2:
                if current not in neural_cache:
                    with torch.no_grad():
                        logits = self.model.get_action_logits(
                            maze,
                            current,
                            visited_history=None,
                            device=self.device
                        )
                        probs = torch.softmax(logits, dim=0)
                        neural_cache[current] = probs.cpu()
                neural_probs = neural_cache[current]

            for action, next_pos, step_cost in neighbors:
                tentative_g = current_g + step_cost
                if next_pos not in g_cost or tentative_g < g_cost[next_pos]:
                    g_cost[next_pos] = tentative_g
                    came_from[next_pos] = current
                    counter += 1

                    # Heuristic Manhattan
                    h_val = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])

                    # Điều chỉnh trọng số bằng Mạng Nơ-ron:
                    # Hướng đi nào được mạng đánh giá cao sẽ được giảm f_score (ưu tiên duyệt trước)
                    prior_bonus = 0.0
                    if neural_probs is not None:
                        dr = next_pos[0] - current[0]
                        dc = next_pos[1] - current[1]
                        act_idx = DELTA_TO_ACTION_IDX.get((dr, dc), 0)
                        prob_score = neural_probs[act_idx].item()
                        # Thưởng điểm định hướng (lên tới 40% trọng số Heuristic)
                        prior_bonus = prob_score * 3.5

                    f_score = tentative_g + max(0.0, h_val - prior_bonus)
                    heapq.heappush(open_set, (f_score, counter, next_pos))

        path = self.reconstruct_path(came_from, goal) if found else []
        total_cost = g_cost.get(goal, float("inf")) if found else float("inf")
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            path=path,
            explored_order=explored_order,
            total_cost=total_cost,
            nodes_explored_count=len(explored_order),
            runtime_ms=elapsed,
            success=found,
        )