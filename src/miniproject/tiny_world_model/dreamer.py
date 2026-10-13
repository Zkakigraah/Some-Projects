import copy
import random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniproject.tiny_world_model.env import GridWorldEnv
from miniproject.tiny_world_model.model import TinyWorldModel


class DreamActorCritic(nn.Module):
    """
    Mạng nơ-ron kết hợp Actor-Critic hoạt động trong không gian biểu diễn của World Model:
    - Actor pi_theta(a | s): Chính sách lựa chọn hành động (Hướng 3).
    - Critic V_phi(s): Hàm ước lượng giá trị tích lũy dài hạn tự học qua TD (Hướng 2).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int = 4,
        embed_dim: int = 48,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.state_emb = nn.Embedding(num_states, embed_dim)

        # Actor Head: Dự đoán logits phân phối xác suất qua 4 hành động
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Critic Head: Dự đoán giá trị vô hướng V(s) của trạng thái
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.state_emb(state_ids)
        action_logits = self.actor(feat)
        values = self.critic(feat)
        return action_logits, values.squeeze(-1)

    def get_action(self, state_idx: int, deterministic: bool = True) -> int:
        """Forward pass O(1) để lấy hành động ngay lập tức ngoài đời thực."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            s_tensor = torch.tensor([state_idx], dtype=torch.long, device=device)
            logits, _ = self.forward(s_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()
        return action


class DreamerTrainer:
    """
    Bộ huấn luyện Policy & Critic hoàn toàn trong trí tưởng tượng của World Model (Dyna / Dreamer paradigm).
    - Sử dụng World Model đã học để unroll các quỹ đạo ảo (Hallucinated Trajectories).
    - Cập nhật Critic V_phi(s) bằng Multi-step Lambda-Return (Dreamer / GAE Paradigm).
    - Cập nhật Actor pi_theta(a|s) bằng Policy Gradient với Normalized Advantage.
    """

    def __init__(
        self,
        world_model: TinyWorldModel,
        actor_critic: DreamActorCritic,
        env: GridWorldEnv,
        gamma: float = 0.95,
        lambda_gae: float = 0.95,
        dream_horizon: int = 8,
        lr: float = 2.5e-3,
        entropy_coef: float = 0.05,
        target_tau: float = 0.05,
        device: str = "cpu",
    ) -> None:
        self.world_model = world_model.to(device)
        self.world_model.eval()  # Freeze World Model trong suốt quá trình dream
        self.ac = actor_critic.to(device)
        self.target_critic = copy.deepcopy(actor_critic).to(device)
        self.target_critic.eval()

        self.env = env
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.dream_horizon = dream_horizon
        self.entropy_coef = entropy_coef
        self.target_tau = target_tau
        self.device = device

        self.optimizer = torch.optim.AdamW(self.ac.parameters(), lr=lr, weight_decay=1e-4)

        # Danh sách các ô tự do hợp lệ để lấy mẫu điểm bắt đầu trong giấc mơ
        self.free_cells = [
            r * env.width + c
            for r in range(env.height)
            for c in range(env.width)
            if (r, c) not in env.obstacles and (r, c) != env.goal_pos
        ]

    def train_dream_step(self, batch_size: int = 32) -> Dict[str, float]:
        self.ac.train()

        # 1. Khởi tạo điểm bắt đầu cho các chuỗi tưởng tượng
        start_states = random.choices(self.free_cells, k=batch_size)

        imagined_states: List[torch.Tensor] = []
        imagined_actions: List[torch.Tensor] = []
        imagined_rewards: List[torch.Tensor] = []
        imagined_dones: List[torch.Tensor] = []
        imagined_next_states: List[torch.Tensor] = []

        curr_states = list(start_states)

        # 2. Unroll đa bước hoàn toàn trong tâm trí của Tiny World Model
        for _ in range(self.dream_horizon):
            s_tensor = torch.tensor(curr_states, dtype=torch.long, device=self.device)
            logits, _ = self.ac(s_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

            next_states_list = []
            rewards_list = []
            dones_list = []

            for s_val, a_val in zip(curr_states, actions.cpu().tolist()):
                pred_s_next, pred_r, pred_d = self.world_model.predict_step(
                    s_val, a_val, deterministic=True
                )
                next_states_list.append(pred_s_next)
                rewards_list.append(pred_r)
                dones_list.append(float(pred_d))

            imagined_states.append(s_tensor)
            imagined_actions.append(actions)
            imagined_rewards.append(torch.tensor(rewards_list, dtype=torch.float32, device=self.device))
            imagined_dones.append(torch.tensor(dones_list, dtype=torch.float32, device=self.device))
            imagined_next_states.append(
                torch.tensor(next_states_list, dtype=torch.long, device=self.device)
            )

            # Đặt lại trạng thái nếu nhánh ảo tưởng đã chạm đích
            curr_states = [
                random.choice(self.free_cells) if d else n_s
                for n_s, d in zip(next_states_list, dones_list)
            ]

        # 3. Tính toán Lambda-Return (R^lambda) truyền ngược thời gian qua Trajectory Horizon
        last_next_s = imagined_next_states[-1]
        with torch.no_grad():
            _, last_v = self.target_critic(last_next_s)

        returns = []
        gae_return = last_v

        # Recursive formulation: R_t = r_t + gamma * (1-d_t) * [ (1-lambda)*V(s_{t+1}) + lambda*R_{t+1} ]
        for t in reversed(range(self.dream_horizon)):
            r_t = imagined_rewards[t]
            d_t = imagined_dones[t]
            s_next_t = imagined_next_states[t]

            with torch.no_grad():
                _, v_next = self.target_critic(s_next_t)

            target_v = (1.0 - self.lambda_gae) * v_next + self.lambda_gae * gae_return
            gae_return = r_t + self.gamma * (1.0 - d_t) * target_v
            returns.insert(0, gae_return)

        # 4. Tối ưu hóa Actor và Critic trên toàn bộ quỹ đạo ảo tưởng
        total_critic_loss = torch.tensor(0.0, device=self.device)
        total_actor_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)

        for t in range(self.dream_horizon):
            s_t = imagined_states[t]
            a_t = imagined_actions[t]
            target_return_t = returns[t]

            logits_t, v_t = self.ac(s_t)
            dist_t = torch.distributions.Categorical(F.softmax(logits_t, dim=-1))
            log_prob_t = dist_t.log_prob(a_t)
            entropy_t = dist_t.entropy().mean()

            # Critic Loss (MSE giữa V(s) và Lambda Target)
            critic_loss = F.mse_loss(v_t, target_return_t)

            # Advantage A(s, a) = R^lambda - V(s)
            advantage = (target_return_t - v_t).detach()

            # Chuẩn hóa Advantage để ổn định Gradient Ascent
            adv_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Actor Loss (Policy Gradient + Entropy Bonus)
            actor_loss = -(log_prob_t * adv_norm).mean() - self.entropy_coef * entropy_t

            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            total_entropy += entropy_t

        total_loss = (total_critic_loss + total_actor_loss) / self.dream_horizon

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Cập nhật mềm (Polyak averaging) mạng Target Critic
        with torch.no_grad():
            for p, p_target in zip(self.ac.parameters(), self.target_critic.parameters()):
                p_target.data.mul_(1.0 - self.target_tau).add_(p.data * self.target_tau)

        return {
            "critic_loss": (total_critic_loss / self.dream_horizon).item(),
            "actor_loss": (total_actor_loss / self.dream_horizon).item(),
            "entropy": (total_entropy / self.dream_horizon).item(),
        }

    def train_in_dreams(self, num_epochs: int = 150, batch_size: int = 48) -> List[Dict[str, float]]:
        """Huấn luyện Agent hàng ngàn bước đi giả tưởng chỉ trong vài giây."""
        history = []
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_dream_step(batch_size=batch_size)
            history.append(metrics)

            if epoch % 30 == 0 or epoch == num_epochs:
                print(
                    f"Dream Epoch {epoch:03d}/{num_epochs:03d} | "
                    f"Critic Loss (TD): {metrics['critic_loss']:.4f} | "
                    f"Actor Loss: {metrics['actor_loss']:.4f} | "
                    f"Policy Entropy: {metrics['entropy']:.3f}"
                )

        return history