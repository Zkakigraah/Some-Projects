import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from miniproject.tiny_world_model.env import GridWorldEnv


class TransitionDataset(Dataset):
    """
    Dataset lưu trữ các bộ chuyển đổi (s_t, a_t, s_{t+1}, r_t, done_t)
    thu được từ việc khám phá môi trường thực.
    """

    def __init__(
        self,
        states: List[int],
        actions: List[int],
        next_states: List[int],
        rewards: List[float],
        dones: List[bool],
    ) -> None:
        self.states = torch.tensor(states, dtype=torch.long)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.next_states = torch.tensor(next_states, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        self.dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "next_state": self.next_states[idx],
            "reward": self.rewards[idx],
            "done": self.dones[idx],
        }

def collect_random_rollouts(
    env: GridWorldEnv, num_episodes: int = 150, max_steps_per_episode: int = 35
) -> TransitionDataset:
    """
    Tập hợp tập dữ liệu huấn luyện bằng chiến lược khám phá ngẫu nhiên (Random Exploration).
    Không yêu cầu agent có trí thông minh trước, chỉ cần khám phá vật lý của thế giới.
    """
    states: List[int] = []
    actions: List[int] = []
    next_states: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []

    total_transitions = 0
    for _ in range(num_episodes):
        # Bắt đầu từ vị trí ngẫu nhiên để phủ kín không gian trạng thái
        s = env.reset(random_start=True)
        for _ in range(max_steps_per_episode):
            a = random.randint(0, len(env.ACTIONS) - 1)
            next_s, r, done, _ = env.step(a)

            states.append(s)
            actions.append(a)
            next_states.append(next_s)
            rewards.append(r)
            dones.append(done)

            total_transitions += 1
            s = next_s
            if done:
                break

    return TransitionDataset(states, actions, next_states, rewards, dones)