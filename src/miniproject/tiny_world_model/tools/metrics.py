from typing import Dict, List
import numpy as np
import torch
from miniproject.tiny_world_model.env import GridWorldEnv
from miniproject.tiny_world_model.model import TinyWorldModel


def evaluate_model_fidelity(
    model: TinyWorldModel,
    env: GridWorldEnv,
    num_test_episodes: int = 50,
    rollout_len: int = 5,
) -> Dict[str, float]:
    """
    Đo lường mức độ tin cậy của World Model khi so sánh chuỗi mô phỏng trong trí tưởng tượng
    với thực tế diễn ra trong môi trường thực qua nhiều bước (1-step, 3-step, 5-step).
    Kiểm tra hiện tượng "ảo giác" (hallucination drift) và hiện tượng đi xuyên tường.
    """
    step_accuracies: List[List[float]] = [[] for _ in range(rollout_len)]
    reward_errors: List[float] = []
    wall_penetration_errors = 0

    for _ in range(num_test_episodes):
        # Thiết lập cùng một trạng thái xuất phát
        real_s = env.reset(random_start=True)
        imagined_s = real_s

        for step_idx in range(rollout_len):
            action = np.random.randint(0, len(env.ACTIONS))

            # 1. Bước đi trong thực tế
            real_next_s, real_r, real_done, real_info = env.step(action)

            # 2. Bước đi trong trí tưởng tượng của World Model
            imagined_next_s, pred_r, pred_done = model.predict_step(
                imagined_s, action, deterministic=True
            )

            # Đánh giá độ trùng khớp vị trí
            is_accurate = float(imagined_next_s == real_next_s)
            step_accuracies[step_idx].append(is_accurate)
            reward_errors.append(abs(pred_r - real_r))

            # Kiểm tra lỗi đi xuyên tường (nếu mô hình dự đoán vị trí chui vào ô tường)
            pred_pos = env.state_idx_to_pos(imagined_next_s)
            if pred_pos in env.obstacles:
                wall_penetration_errors += 1

            # Dùng tiếp trạng thái tưởng tượng cho bước kế tiếp để đo lường sai số trôi dạt (autoregressive drift)
            imagined_s = imagined_next_s

            if real_done:
                break

    results: Dict[str, float] = {
        "1_step_accuracy": float(np.mean(step_accuracies[0])),
        "3_step_accuracy": float(np.mean(step_accuracies[min(2, rollout_len - 1)])),
        "multi_step_avg_accuracy": float(
            np.mean([np.mean(accs) for accs in step_accuracies if len(accs) > 0])
        ),
        "mean_reward_mae": float(np.mean(reward_errors)),
        "wall_penetration_count": wall_penetration_errors,
    }

    return results