from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from miniproject.tiny_world_model.dataset import TransitionDataset
from miniproject.tiny_world_model.model import TinyWorldModel


class WorldModelTrainer:
    """
    Quản lý quá trình huấn luyện World Model, cân bằng hàm mất mát hỗn hợp:
    L = L_transition (CrossEntropy) + alpha * L_reward (MSE) + beta * L_done (BCE)
    """

    def __init__(
        self,
        model: TinyWorldModel,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        alpha_reward: float = 2.0,
        beta_done: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.alpha_reward = alpha_reward
        self.beta_done = beta_done

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.mse_loss_fn = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        state_acc_total = 0.0
        reward_loss_total = 0.0
        count = 0

        for batch in dataloader:
            s = batch["state"].to(self.device)
            a = batch["action"].to(self.device)
            target_next_s = batch["next_state"].to(self.device)
            target_r = batch["reward"].to(self.device)
            target_done = batch["done"].to(self.device)

            self.optimizer.zero_grad()
            next_state_logits, pred_r, pred_done_logit = self.model(s, a)

            # Loss dự đoán trạng thái tiếp theo (Classification)
            loss_state = F.cross_entropy(next_state_logits, target_next_s)

            # Loss dự đoán phần thưởng (Regression)
            loss_reward = self.mse_loss_fn(pred_r, target_r)

            # Loss dự đoán điều kiện kết thúc (Binary Classification)
            loss_done = self.bce_loss_fn(pred_done_logit, target_done)

            combined_loss = loss_state + self.alpha_reward * loss_reward + self.beta_done * loss_done
            combined_loss.backward()
            self.optimizer.step()

            # Độ chính xác phân loại trạng thái kế tiếp
            with torch.no_grad():
                preds = torch.argmax(next_state_logits, dim=-1)
                acc = (preds == target_next_s).float().mean().item()

            batch_size = s.size(0)
            total_loss += combined_loss.item() * batch_size
            state_acc_total += acc * batch_size
            reward_loss_total += loss_reward.item() * batch_size
            count += batch_size

        return {
            "loss": total_loss / count,
            "state_acc": state_acc_total / count,
            "reward_mse": reward_loss_total / count,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        state_acc_total = 0.0
        count = 0

        for batch in dataloader:
            s = batch["state"].to(self.device)
            a = batch["action"].to(self.device)
            target_next_s = batch["next_state"].to(self.device)
            target_r = batch["reward"].to(self.device)
            target_done = batch["done"].to(self.device)

            next_state_logits, pred_r, pred_done_logit = self.model(s, a)
            loss_state = F.cross_entropy(next_state_logits, target_next_s)
            loss_reward = self.mse_loss_fn(pred_r, target_r)
            loss_done = self.bce_loss_fn(pred_done_logit, target_done)

            combined_loss = loss_state + self.alpha_reward * loss_reward + self.beta_done * loss_done
            preds = torch.argmax(next_state_logits, dim=-1)
            acc = (preds == target_next_s).float().mean().item()

            batch_size = s.size(0)
            total_loss += combined_loss.item() * batch_size
            state_acc_total += acc * batch_size
            count += batch_size

        return {
            "val_loss": total_loss / count,
            "val_state_acc": state_acc_total / count,
        }

    def fit(
        self,
        dataset: TransitionDataset,
        epochs: int = 35,
        batch_size: int = 64,
        val_ratio: float = 0.15,
    ) -> List[Dict[str, float]]:
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            log = {**train_metrics, **val_metrics, "epoch": epoch}
            history.append(log)

            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {log['loss']:.4f} | "
                    f"Acc (State): {log['state_acc'] * 100:.2f}% | "
                    f"Val Acc: {log['val_state_acc'] * 100:.2f}%"
                )

        return history