import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

from miniproject.maze.grid import Coordinate, Action
from miniproject.maze.fog import FogOfWarEnvironment, FogCellState


@dataclass
class OnlineSearchResult:
    """Kết quả giải mê cung trong điều kiện sương mù (POMDP)."""
    algorithm_name: str
    path_taken: List[Coordinate]
    total_steps: int
    replanning_calls: int
    coverage_percent: float
    runtime_ms: float
    success: bool


class OnlineBaseSolver(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(
        self,
        env: FogOfWarEnvironment,
        max_steps: int = 4000,
        animate: bool = False,
        animate_delay: float = 0.04
    ) -> OnlineSearchResult:
        pass


class OptimisticReplanningAStar(OnlineBaseSolver):
    """
    Online Replanning with Free-Space Assumption (Chuẩn mực trong Robotics):
    - Nguyên lý: Lạc quan giả định tất cả các ô sương mù '?' đều là đường đi trống.
    - Agent dùng A* vạch lộ trình giả định từ vị trí hiện tại tới Goal.
    - Agent tiến bước theo lộ trình đó.
    - Bất cứ khi nào cảm biến quét thấy một bức tường mới mọc ra chắn ngang lộ trình đã định:
      -> Ngay lập tức kích hoạt Tái lập kế hoạch (Replanning)!
    """

    def __init__(self):
        super().__init__("Optimistic Online A* (Free-Space Assumption)")

    @staticmethod
    def _plan_optimistic_path(
        internal_map: List[List[str]],
        start: Coordinate,
        goal: Coordinate,
        height: int,
        width: int
    ) -> Optional[List[Coordinate]]:
        """Lập kế hoạch A* trên bản đồ niềm tin nội bộ."""
        if start == goal:
            return [start]

        counter = 0
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        open_set: List[Tuple[float, float, int, Coordinate]] = [(h_start, 0.0, counter, start)]
        came_from: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        g_cost: Dict[Coordinate, float] = {start: 0.0}
        closed_set: Set[Coordinate] = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, current_g, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                # Tái dựng đường đi
                path = []
                curr: Optional[Coordinate] = current
                while curr is not None:
                    path.append(curr)
                    curr = came_from.get(curr)
                path.reverse()
                return path

            cr, cc = current
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width:
                    cell_type = internal_map[nr][nc]
                    # Nguyên lý Free-Space: '?' (sương mù) được coi là có thể đi qua!
                    if cell_type != FogCellState.WALL:
                        tentative_g = current_g + 1.0
                        nxt = (nr, nc)
                        if nxt not in g_cost or tentative_g < g_cost[nxt]:
                            g_cost[nxt] = tentative_g
                            came_from[nxt] = current
                            counter += 1
                            h_val = abs(nr - goal[0]) + abs(nc - goal[1])
                            heapq.heappush(open_set, (tentative_g + h_val, tentative_g, counter, nxt))

        return None

    def solve(
        self,
        env: FogOfWarEnvironment,
        max_steps: int = 4000,
        animate: bool = False,
        animate_delay: float = 0.03
    ) -> OnlineSearchResult:
        start_time = time.perf_counter()
        env.reset()

        goal = env.goal_pos
        height, width = env.height, env.width
        replanning_calls = 0
        planned_path: List[Coordinate] = []

        step_count = 0
        success = False

        while step_count < max_steps:
            if env.agent_pos == goal:
                success = True
                break

            # Kiểm tra xem đường đi đã vạch sẵn có còn khả thi hay bị tường mới phát hiện chặn đứng
            path_blocked = False
            if planned_path:
                # Kiểm tra các ô tiếp theo trong lộ trình xem có ô nào bị phát hiện là WALL không
                for cell in planned_path:
                    if env.internal_map[cell[0]][cell[1]] == FogCellState.WALL:
                        path_blocked = True
                        break

            # Nếu chưa có lộ trình hoặc lộ trình bị tường bít -> REPLAN!
            if not planned_path or path_blocked:
                replanning_calls += 1
                full_path = self._plan_optimistic_path(
                    env.internal_map,
                    env.agent_pos,
                    goal,
                    height,
                    width
                )
                if full_path is None or len(full_path) < 2:
                    # Hoàn toàn không còn đường tới đích
                    break
                planned_path = full_path[1:]  # Bỏ vị trí hiện tại

            # Tiến bước tiếp theo dọc theo kế hoạch
            next_step = planned_path.pop(0)
            dr = next_step[0] - env.agent_pos[0]
            dc = next_step[1] - env.agent_pos[1]

            # Ánh xạ vector sang Action
            act = None
            for a in Action:
                if a.delta == (dr, dc):
                    act = a
                    break
            assert act is not None

            # Thực thi bước đi trong môi trường sương mù
            obs, reward, done, info = env.step(act)
            step_count += 1

            if animate:
                import os
                os.system("cls" if os.name == "nt" else "clear")
                print(f"--- FOG OF WAR: Bước {step_count} | Số lần Replan: {replanning_calls} | Độ phủ: {env.get_discovered_coverage():.1f}% ---\n")
                print(env.render_ansi())
                time.sleep(animate_delay)

            if done:
                success = True
                break

        elapsed = (time.perf_counter() - start_time) * 1000.0

        return OnlineSearchResult(
            algorithm_name=self.name,
            path_taken=list(env.visited_history),
            total_steps=len(env.visited_history) - 1,
            replanning_calls=replanning_calls,
            coverage_percent=env.get_discovered_coverage(),
            runtime_ms=elapsed,
            success=success,
        )


class OnlineSensorTremauxSolver(OnlineBaseSolver):
    """
    Thuật toán khám phá cảm biến cục bộ Trémaux:
    - Thuần túy dựa vào cảm biến cửa sổ 3x3 và đánh dấu số lần đi qua (Marking Heuristic).
    - Không cần biết tọa độ đích trước, tự dò dẫm trong sương mù để tìm Goal.
    """

    def __init__(self):
        super().__init__("Sensor-Based Trémaux Exploration")

    def solve(
        self,
        env: FogOfWarEnvironment,
        max_steps: int = 4000,
        animate: bool = False,
        animate_delay: float = 0.03
    ) -> OnlineSearchResult:
        start_time = time.perf_counter()
        env.reset()

        visit_counts: Dict[Coordinate, int] = {env.agent_pos: 1}
        step_count = 0
        success = False

        while step_count < max_steps:
            if env.agent_pos == env.goal_pos:
                success = True
                break

            curr = env.agent_pos
            # Kiểm tra 4 hướng lân cận trong bản đồ nội tâm
            valid_moves = []
            for action in Action:
                dr, dc = action.delta
                nxt = (curr[0] + dr, curr[1] + dc)
                if 0 <= nxt[0] < env.height and 0 <= nxt[1] < env.width:
                    if env.internal_map[nxt[0]][nxt[1]] != FogCellState.WALL:
                        # Ưu tiên các ô chưa từng bước tới
                        count = visit_counts.get(nxt, 0)
                        # Heuristic hướng về đích nếu đích đã biết
                        dist_h = abs(nxt[0] - env.goal_pos[0]) + abs(nxt[1] - env.goal_pos[1])
                        valid_moves.append((count, dist_h, action, nxt))

            if not valid_moves:
                break

            # Chọn bước đi có số lần viếng thăm ít nhất (ưu tiên ô mới)
            valid_moves.sort(key=lambda item: (item[0], item[1]))
            _, _, chosen_act, nxt_cell = valid_moves[0]

            obs, reward, done, info = env.step(chosen_act)
            visit_counts[env.agent_pos] = visit_counts.get(env.agent_pos, 0) + 1
            step_count += 1

            if animate:
                import os
                os.system("cls" if os.name == "nt" else "clear")
                print(f"--- TRÉMAUX KHÁM PHÁ: Bước {step_count} | Độ phủ: {env.get_discovered_coverage():.1f}% ---\n")
                print(env.render_ansi())
                time.sleep(animate_delay)

            if done:
                success = True
                break

        elapsed = (time.perf_counter() - start_time) * 1000.0

        return OnlineSearchResult(
            algorithm_name=self.name,
            path_taken=list(env.visited_history),
            total_steps=len(env.visited_history) - 1,
            replanning_calls=0,
            coverage_percent=env.get_discovered_coverage(),
            runtime_ms=elapsed,
            success=success,
        )