import math
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from miniproject.maze.grid import Maze, Action, Coordinate
from miniproject.maze.solvers import AStarSolver

DELTA_TO_ACTION_IDX = {
    (-1, 0): 0,  # UP
    (1, 0): 1,   # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,   # RIGHT
}

ACTION_IDX_TO_ACTION = [
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
]


def compute_goal_vector(current_pos: Coordinate, goal: Coordinate) -> torch.Tensor:
    """Tính vector đơn vị chuẩn hóa (Compass Vector) từ vị trí Agent đến Goal."""
    dr = float(goal[0] - current_pos[0])
    dc = float(goal[1] - current_pos[1])
    dist = math.hypot(dr, dc)
    if dist > 1e-6:
        return torch.tensor([dr / dist, dc / dist], dtype=torch.float32)
    return torch.tensor([0.0, 0.0], dtype=torch.float32)


def extract_local_patch(
    maze: Maze,
    current_pos: Coordinate,
    visited_history: Optional[List[Coordinate]] = None,
    patch_size: int = 15
) -> torch.Tensor:
    """
    Trích xuất cửa sổ cục bộ (Egocentric Patch) 15x15 quanh Agent:
    - Kênh 0: Bản đồ Tường trong tầm nhìn (1.0 nếu là tường hoặc vượt ra ngoài biên mê cung).
    - Kênh 1: Dấu chân lịch sử (Footprint Trail) suy giảm theo thời gian.
    - Kênh 2: Vị trí Goal (1.0 nếu Goal nằm trong tầm nhìn cục bộ, ngược lại 0.0).
    """
    radius = patch_size // 2
    cur_r, cur_c = current_pos
    patch = torch.zeros((3, patch_size, patch_size), dtype=torch.float32)

    # 1. Kênh Tường chắn
    for i in range(patch_size):
        r = cur_r - radius + i
        for j in range(patch_size):
            c = cur_c - radius + j
            if not maze.is_traversable((r, c)):
                patch[0, i, j] = 1.0

    # 2. Kênh Dấu chân (Visited Trail)
    if visited_history:
        recent = visited_history[-15:]
        num_recent = len(recent)
        for idx, (vr, vc) in enumerate(recent):
            pi = vr - (cur_r - radius)
            pj = vc - (cur_c - radius)
            if 0 <= pi < patch_size and 0 <= pj < patch_size:
                decay = 0.85 ** (num_recent - 1 - idx)
                patch[1, pi, pj] = max(patch[1, pi, pj].item(), float(decay))

    # 3. Kênh Goal
    if maze.goal is not None:
        gr, gc = maze.goal
        gi = gr - (cur_r - radius)
        gj = gc - (cur_c - radius)
        if 0 <= gi < patch_size and 0 <= gj < patch_size:
            patch[2, gi, gj] = 1.0

    return patch


def augment_sample(
    patch: torch.Tensor,
    goal_vec: torch.Tensor,
    action_idx: int
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Áp dụng nhóm đối xứng D4 (4 góc quay x 2 hướng lật gương) để nhân 8 dữ liệu:
    Bảo toàn tính bất biến không gian của góc cua và hướng di chuyển.
    """
    samples = []
    # Vector chuyển vị cơ bản cho 4 action [UP, DOWN, LEFT, RIGHT]
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for k in range(4):  # Xoay 0, 90, 180, 270 độ
        rot_patch = torch.rot90(patch, k=k, dims=[1, 2])
        # Xoay vector Goal và Action tương ứng theo k góc 90 độ theo chiều kim đồng hồ
        # Với k=1: (dr, dc) -> (dc, -dr)
        g_dr, g_dc = goal_vec[0].item(), goal_vec[1].item()
        act_dr, act_dc = deltas[action_idx]

        for _ in range(k):
            g_dr, g_dc = g_dc, -g_dr
            act_dr, act_dc = act_dc, -act_dr

        rot_g_vec = torch.tensor([g_dr, g_dc], dtype=torch.float32)
        new_act = DELTA_TO_ACTION_IDX.get((int(round(act_dr)), int(round(act_dc))), action_idx)
        samples.append((rot_patch, rot_g_vec, new_act))

        # Lật ngang (Horizontal Flip)
        flip_patch = torch.flip(rot_patch, dims=[2])
        flip_g_vec = torch.tensor([g_dr, -g_dc], dtype=torch.float32)
        flip_act_dr, flip_act_dc = act_dr, -act_dc
        flip_act = DELTA_TO_ACTION_IDX.get((int(round(flip_act_dr)), int(round(flip_act_dc))), new_act)
        samples.append((flip_patch, flip_g_vec, flip_act))

    return samples


class MazeTrajectoryDataset(Dataset):
    """
    Dataset chứa bộ ba: (Local Patch 15x15, Goal Vector 2D, Action)
    được tăng cường 8 lần kích thước bằng Data Augmentation.
    """

    def __init__(self, mazes: List[Maze]):
        self.patches: List[torch.Tensor] = []
        self.goal_vectors: List[torch.Tensor] = []
        self.actions: List[int] = []

        solver = AStarSolver()

        for maze in mazes:
            res = solver.solve(maze)
            if not res.success or len(res.path) < 2:
                continue

            assert maze.goal is not None
            history: List[Coordinate] = []

            for t in range(len(res.path) - 1):
                cur_pos = res.path[t]
                next_pos = res.path[t + 1]
                history.append(cur_pos)

                dr = next_pos[0] - cur_pos[0]
                dc = next_pos[1] - cur_pos[1]
                action_idx = DELTA_TO_ACTION_IDX.get((dr, dc))

                if action_idx is not None:
                    raw_patch = extract_local_patch(maze, cur_pos, history, patch_size=15)
                    raw_g_vec = compute_goal_vector(cur_pos, maze.goal)

                    # Tăng cường dữ liệu bằng nhóm đối xứng D4
                    augmented = augment_sample(raw_patch, raw_g_vec, action_idx)
                    for p, g, a in augmented:
                        self.patches.append(p)
                        self.goal_vectors.append(g)
                        self.actions.append(a)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.patches[idx], self.goal_vectors[idx], self.actions[idx]