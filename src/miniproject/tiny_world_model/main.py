import random
import time
import torch
import numpy as np

from miniproject.tiny_world_model.env import GridWorldEnv
from miniproject.tiny_world_model.model import TinyWorldModel
from miniproject.tiny_world_model.dataset import collect_random_rollouts
from miniproject.tiny_world_model.trainer import WorldModelTrainer
from miniproject.tiny_world_model.dreamer import DreamActorCritic, DreamerTrainer
from miniproject.tiny_world_model.tools.visualizer import TerminalVisualizer
from miniproject.tiny_world_model.tools.metrics import evaluate_model_fidelity


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== TINY WORLD MODEL: DYNA-DREAMER PIPELINE ===")
    print(f"Using Compute Device: {device}\n")

    # BƯỚC 1: Khởi tạo thế giới thực (GridWorld)
    env = GridWorldEnv(height=6, width=6, start_pos=(0, 0), goal_pos=(5, 5))
    print(TerminalVisualizer.render_grid(env, title="Môi trường thực tế (Real Environment)"))
    print("Mô tả: [A] Agent, [G] Mục tiêu, [#] Chướng ngại vật, [.] Ô trống tự do.\n")

    # BƯỚC 2: Thu thập dữ liệu bằng hành vi ngẫu nhiên
    print(">> Đang thu thập dữ liệu chuyển đổi trạng thái bằng Random Exploration...")
    dataset = collect_random_rollouts(env, num_episodes=220, max_steps_per_episode=35)
    print(f"Thu thập thành công {len(dataset)} mẫu chuyển đổi (s, a, s', r, done).\n")

    # BƯỚC 3: Xây dựng và Huấn luyện World Model f(s, a) -> (s', r, done)
    print(">> Khởi tạo và huấn luyện Tiny World Model...")
    world_model = TinyWorldModel(
        num_states=env.num_states, num_actions=len(env.ACTIONS), embed_dim=64, hidden_dim=128
    )
    trainer = WorldModelTrainer(world_model, learning_rate=3e-3, device=device)
    trainer.fit(dataset, epochs=25, batch_size=64)
    print(">> Huấn luyện World Model hoàn tất!\n")

    # BƯỚC 4: Đo lường độ tin cậy mô phỏng
    metrics = evaluate_model_fidelity(world_model, env, num_test_episodes=60, rollout_len=5)
    print(f">> Độ tin cậy World Model: 1-Step: {metrics['1_step_accuracy'] * 100:.2f}% | Wall Error: {metrics['wall_penetration_count']}")

    # BƯỚC 5: HƯỚNG 2 & 3 - HUẤN LUYỆN ACTOR-CRITIC TRONG GIẤC MƠ (DREAM TRAINING)
    print("\n" + "=" * 65)
    print(">> BẮT ĐẦU PHA HUẤN LUYỆN CHÍNH SÁCH TRONG GIẤC MƠ (DREAMING PHASE)...")
    print("   * Không dùng Heuristic Manhattan hay tọa độ con người cung cấp.")
    print("   * Critic V(s) tự học thế năng bằng Bellman TD-Learning (Hướng 2).")
    print("   * Actor pi(a|s) học chọn hướng tối ưu bằng Policy Gradient (Hướng 3).")
    print("=" * 65)

    actor_critic = DreamActorCritic(
        num_states=env.num_states, num_actions=len(env.ACTIONS), embed_dim=48, hidden_dim=64
    )
    dream_trainer = DreamerTrainer(
        world_model=world_model,
        actor_critic=actor_critic,
        env=env,
        gamma=0.92,
        dream_horizon=6,
        lr=3e-3,
        entropy_coef=0.04,
        device=device,
    )

    t_start = time.time()
    dream_trainer.train_in_dreams(num_epochs=120, batch_size=64)
    print(f">> Huấn luyện trong giấc mơ hoàn tất chỉ sau {time.time() - t_start:.2f} giây!\n")

    # BƯỚC 6: TRỰC QUAN HÓA TRI THỨC AGENT TỰ KHÁM PHÁ
    print(TerminalVisualizer.render_value_heatmap(env, actor_critic))
    print("\n" + TerminalVisualizer.render_policy_field(env, actor_critic) + "\n")

    # BƯỚC 7: THỰC THI THẾ GIỚI THỰC VỚI ĐỘ TRỄ O(1) (ZERO-SEARCH LATENCY)
    print(">> TIẾN HÀNH GIẢI THỰC TẾ BẰNG FORWARD PASS TRỰC TIẾP O(1)...")
    curr_state = env.reset(random_start=False)
    actual_path = [env.state_idx_to_pos(curr_state)]
    solved = False

    for step in range(1, 25):
        # O(1) Inference: Lấy hành động trực tiếp từ Actor đã tôi luyện trong mơ
        t0 = time.perf_counter()
        chosen_action = actor_critic.get_action(curr_state, deterministic=True)
        decision_time_ms = (time.perf_counter() - t0) * 1000

        next_state, reward, done, info = env.step(chosen_action)
        actual_path.append(info["agent_pos"])
        curr_state = next_state

        print(
            f"Bước {step:02d} | Vị trí: {str(info['agent_pos']):<8} | "
            f"Hành động: {env.ACTION_NAMES[chosen_action]:<5} | "
            f"Reward: {reward:5.1f} | Latency: {decision_time_ms:.4f} ms"
        )

        if info["reached_goal"]:
            print(f"\n🎉 THÀNH CÔNG! Agent đã về đích chỉ trong {step} bước với độ trễ mỗi quyết định gần như bằng 0!")
            solved = True
            break

    print("\n" + TerminalVisualizer.render_grid(
        env, agent_pos=env.agent_pos, path=actual_path, title="Lộ trình thực tế được tạo bởi Dream Policy"
    ))
    if not solved:
        print("Không tới được đích trong số bước quy định.")


if __name__ == "__main__":
    main()