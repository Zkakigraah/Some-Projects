from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyWorldModel(nn.Module):
    """
    World Model xấp xỉ hàm chuyển đổi động học:
        f(s_t, a_t) -> (s_{t+1}, r_t, done_t)

    Kiến trúc gồm:
    - State Embedding & Action Embedding
    - Shared Representation Trunk (MLP với LayerNorm)
    - Next-State Prediction Head (Phân loại xác suất qua toàn bộ cell)
    - Reward Prediction Head (Hồi quy giá trị scalar)
    - Done/Terminal Head (Phân loại nhị phân cờ kết thúc)
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.state_emb = nn.Embedding(num_states, embed_dim)
        self.action_emb = nn.Embedding(num_actions, embed_dim)

        input_dim = embed_dim * 2
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 1. Dự đoán next state: Logits phân bố trên num_states
        self.next_state_head = nn.Linear(hidden_dim, num_states)

        # 2. Dự đoán reward: Giá trị vô hướng scalar
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 3. Dự đoán done: Logit nhị phân
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, state_ids: torch.Tensor, action_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass tính toán đồng thời các đại lượng.
        Args:
            state_ids: Tensor [B] chỉ số trạng thái
            action_ids: Tensor [B] chỉ số hành động
        Returns:
            next_state_logits: [B, num_states]
            reward_pred: [B, 1]
            done_logit: [B, 1]
        """
        s_feat = self.state_emb(state_ids)
        a_feat = self.action_emb(action_ids)

        x = torch.cat([s_feat, a_feat], dim=-1)
        latent = self.trunk(x)

        next_state_logits = self.next_state_head(latent)
        reward_pred = self.reward_head(latent)
        done_logit = self.done_head(latent)

        return next_state_logits, reward_pred, done_logit

    @torch.no_grad()
    def predict_step(
        self, state_idx: int, action: int, deterministic: bool = True
    ) -> Tuple[int, float, bool]:
        """
        Mô phỏng 1 bước chuyển đổi trạng thái hoàn toàn trong trí tưởng tượng (In-Model Step).
        Không tương tác với môi trường thật.
        """
        self.eval()
        device = next(self.parameters()).device
        s_tensor = torch.tensor([state_idx], dtype=torch.long, device=device)
        a_tensor = torch.tensor([action], dtype=torch.long, device=device)

        next_state_logits, reward_pred, done_logit = self(s_tensor, a_tensor)

        if deterministic:
            next_state = torch.argmax(next_state_logits, dim=-1).item()
        else:
            probs = F.softmax(next_state_logits, dim=-1)
            next_state = torch.multinomial(probs, num_samples=1).item()

        reward = reward_pred.item()
        done = torch.sigmoid(done_logit).item() > 0.5

        return next_state, reward, done