import time
from pathlib import Path
from typing import List, Tuple, Set
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miniproject.maze.grid import Maze, Coordinate
from miniproject.maze.generator import MazeGenerator
from miniproject.maze.dataset import MazeTrajectoryDataset
from miniproject.maze.model import MazePolicyCNN
from miniproject.maze.solvers import NeuralPolicySolver
from miniproject.maze.visualizer import TerminalColors

CHECKPOINT_PATH = Path(__file__).resolve().parent / "maze_policy.pth"


def train_policy_network(
    model: MazePolicyCNN,
    dataloader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{TerminalColors.BOLD}🚀 BẮT ĐẦU HUẤN LUYỆN MULTI-SCALE POLICY (15x15 + DATA AUGMENTATION)...{TerminalColors.RESET}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for patches, goal_vecs, targets in dataloader:
            patches = patches.to(device)
            goal_vecs = goal_vecs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(patches, goal_vecs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * patches.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += patches.size(0)

        scheduler.step()
        epoch_loss = total_loss / total
        accuracy = (correct / total) * 100.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")


def evaluate_policy_generalization(
    model: MazePolicyCNN,
    train_mazes: List[Maze],
    test_mazes: List[Maze],
    device: torch.device
) -> None:
    print(f"\n{'='*75}")
    print(f"{TerminalColors.BOLD}{TerminalColors.CYAN}🔬 KIỂM CHỨNG TỔNG QUÁT HÓA CỦA MÔ HÌNH HIERARCHICAL AGENT{TerminalColors.RESET}")
    print(f"{'='*75}")

    solver = NeuralPolicySolver(model_path=str(CHECKPOINT_PATH))
    solver.model = model
    solver.device = device

    def eval_suite(mazes: List[Maze], suite_name: str):
        successes = 0
        runtimes = []

        for m in mazes:
            res = solver.solve(m)
            if res.success:
                successes += 1
            runtimes.append(res.runtime_ms)

        rate = (successes / len(mazes)) * 100.0
        avg_time = sum(runtimes) / len(runtimes) if runtimes else 0.0
        opt_color = TerminalColors.GREEN if rate > 90 else TerminalColors.YELLOW
        print(f"\n{TerminalColors.BOLD}[{suite_name}] (Tổng {len(mazes)} mê cung):{TerminalColors.RESET}")
        print(f"-> Tỷ lệ giải thành công: {opt_color}{rate:.1f}%{TerminalColors.RESET}")
        print(f"-> Thời gian giải trung bình: {avg_time:.2f} ms")

    eval_suite(train_mazes, "TẬP TRAIN (Kích thước 15x15)")
    eval_suite(test_mazes, "TẬP TEST LỚN (Kích thước 25x25)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    num_train = 120
    num_test = 40

    print(f"Đang sinh {num_train} mê cung Train (15x15) và {num_test} mê cung Test lớn (25x25)...")
    train_mazes = [MazeGenerator.generate(15, 15, loop_factor=0.08, seed=i) for i in range(num_train)]
    test_mazes = [MazeGenerator.generate(25, 25, loop_factor=0.08, seed=8000 + i) for i in range(num_test)]

    print("Đang trích xuất quỹ đạo A* kết hợp Data Augmentation 8 hướng (D4 Symmetry)...")
    dataset = MazeTrajectoryDataset(train_mazes)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Tổng số mẫu huấn luyện thu được sau Data Augmentation: {len(dataset)}")

    model = MazePolicyCNN(in_channels=3, num_actions=4)
    train_policy_network(model, dataloader, epochs=15, lr=0.001, device=device)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(CHECKPOINT_PATH))
    print(f"\n{TerminalColors.GREEN}✓ Đã lưu trọng số mô hình vào: {CHECKPOINT_PATH.resolve()}{TerminalColors.RESET}")

    evaluate_policy_generalization(model, train_mazes, test_mazes, device)


if __name__ == "__main__":
    main()