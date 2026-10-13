import torch
import torch.nn as nn
from typing import Optional, List, Dict
from miniproject.maze.grid import Maze, Coordinate, Action
from miniproject.maze.dataset import (
    extract_local_patch,
    compute_goal_vector,
    ACTION_IDX_TO_ACTION,
)


class MazePolicyCNN(nn.Module):
    """
    Mạng nơ-ron cục bộ (Egocentric Policy Network):
    - Nhận Patch cục bộ 15x15 (Tường, Dấu chân, Goal gần)
    - Nhận Goal Compass Vector (dr/d, dc/d) định hướng toàn cục
    - Dự đoán phân phối xác suất trên 4 hành động: UP, DOWN, LEFT, RIGHT.
    """

    def __init__(self, in_channels: int = 3, num_actions: int = 4):
        super().__init__()

        # Khối trích xuất đặc trưng không gian trên patch 15x15
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 7, 7)

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        flatten_dim = 64 * 7 * 7  # 3136

        # Classifier kết hợp đặc trưng hình thái học và vector hướng Goal
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(128, num_actions),
        )

    def forward(self, patch: torch.Tensor, goal_vec: torch.Tensor) -> torch.Tensor:
        """
        Input:
          patch: (B, 3, 15, 15)
          goal_vec: (B, 2)
        Output:
          logits: (B, 4)
        """
        feats = self.conv(patch)
        feats_flat = torch.flatten(feats, start_dim=1)
        combined = torch.cat([feats_flat, goal_vec], dim=1)
        return self.classifier(combined)

    def get_action_logits(
        self,
        maze: Maze,
        current_pos: Coordinate,
        visited_history: Optional[List[Coordinate]] = None,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Tính toán phân phối xác suất/logits của 4 hướng đi."""
        self.eval()
        with torch.no_grad():
            assert maze.goal is not None
            patch = extract_local_patch(maze, current_pos, visited_history=visited_history, patch_size=15)
            patch_tensor = patch.unsqueeze(0).to(device)

            g_vec = compute_goal_vector(current_pos, maze.goal)
            g_vec_tensor = g_vec.unsqueeze(0).to(device)

            logits = self.forward(patch_tensor, g_vec_tensor).squeeze(0)
            return logits

    def predict_action(
        self,
        maze: Maze,
        current_pos: Coordinate,
        visited_history: Optional[List[Coordinate]] = None,
        device: torch.device = torch.device("cpu"),
        filter_invalid: bool = True
    ) -> Action:
        """Dự đoán hành động tốt nhất, tự động lọc các bước đâm đầu vào tường."""
        logits = self.get_action_logits(maze, current_pos, visited_history, device)

        if filter_invalid:
            for idx, action in enumerate(ACTION_IDX_TO_ACTION):
                dr, dc = action.delta
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)
                if not maze.is_traversable(next_pos):
                    logits[idx] = -1e9

        action_idx = int(torch.argmax(logits).item())
        return ACTION_IDX_TO_ACTION[action_idx]