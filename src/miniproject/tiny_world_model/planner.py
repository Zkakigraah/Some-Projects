import itertools
from typing import Dict, List, Optional, Set, Tuple
from miniproject.tiny_world_model.model import TinyWorldModel


class ImaginedRolloutPlanner:
    """
    Model-Based Planner hoạt động bằng cách mô phỏng tương lai trong trí tưởng tượng của World Model.
    
    Được nâng cấp để giải quyết bài toán Myopic Horizon (Tầm nhìn thiển cận):
    1. Terminal Value Estimation (Heuristic potential hướng về mục tiêu)
    2. Recency Penalty (Chống dao động lặp qua lại giữa các ô kề nhau)
    """

    def __init__(
        self,
        world_model: TinyWorldModel,
        planning_horizon: int = 4,
        discount_gamma: float = 0.9,
        goal_pos: Optional[Tuple[int, int]] = None,
        grid_width: int = 6,
        heuristic_weight: float = 1.2,
        recency_penalty: float = 0.4,
    ) -> None:
        self.world_model = world_model
        self.planning_horizon = planning_horizon
        self.gamma = discount_gamma
        self.goal_pos = goal_pos
        self.grid_width = grid_width
        self.heuristic_weight = heuristic_weight
        self.recency_penalty = recency_penalty

        # Lịch sử các bước thực tế gần đây để ngăn chặn hiện tượng kẹt dao động
        self.recent_visited_states: List[int] = []

    def compute_terminal_value(self, state_idx: int) -> float:
        """
        Ước lượng giá trị dài hạn V(s) của trạng thái lá cuối chuỗi mô phỏng.
        Trong GridWorld, khoảng cách Manhattan nghịch đảo là hàm Heuristic Admissible 
        đóng vai trò như một Value Function định hướng dốc (gradient) về mục tiêu.
        """
        if self.goal_pos is None:
            return 0.0

        r = state_idx // self.grid_width
        c = state_idx % self.grid_width
        manhattan_dist = abs(r - self.goal_pos[0]) + abs(c - self.goal_pos[1])
        
        # Càng gần đích khoảng cách càng nhỏ -> Giá trị tiềm năng càng cao (ít âm hơn)
        return -float(manhattan_dist) * self.heuristic_weight

    def simulate_trajectory(
        self,
        start_state: int,
        action_sequence: Tuple[int, ...],
        recent_history: Optional[List[int]] = None,
    ) -> Tuple[float, List[int]]:
        """
        Rollout một chuỗi hành động trong đầu mô hình:
        s0 -> (a0) -> s1 -> (a1) -> s2 ...
        Đồng thời kết hợp phần thưởng tích lũy, phạt lặp và Terminal Value tại lá.
        """
        curr_state = start_state
        total_return = 0.0
        current_discount = 1.0
        imagined_path = [curr_state]

        history_set: Set[int] = set(recent_history[-3:]) if recent_history else set()

        for step_idx, a in enumerate(action_sequence):
            next_state, reward, done = self.world_model.predict_step(
                curr_state, a, deterministic=True
            )

            # Phạt nhẹ nếu nhánh tưởng tượng cố tình quay lại ô vừa mới đi qua
            step_reward = reward
            if next_state in history_set and step_idx == 0:
                step_reward -= self.recency_penalty

            total_return += current_discount * step_reward
            imagined_path.append(next_state)
            curr_state = next_state

            if done:
                # Nếu đạt đích trong trí tưởng tượng, nhận toàn bộ giá trị cực đại
                break
            current_discount *= self.gamma

        # Bổ sung Terminal Value Function V(s_H) vào trạng thái kết thúc nhánh
        terminal_value = self.compute_terminal_value(curr_state)
        total_return += current_discount * terminal_value

        return total_return, imagined_path

    def plan_next_action(
        self, current_state: int
    ) -> Tuple[int, Dict[int, Dict[str, object]]]:
        """
        Mô phỏng toàn bộ các nhánh hành động khả dĩ, đánh giá Cumulative Return + Terminal Value.
        Returns:
            best_action: Hành động có tiềm năng cao nhất
            branch_analysis: Thông tin chi tiết phục vụ trực quan hóa
        """
        num_actions = self.world_model.num_actions
        branch_analysis: Dict[int, Dict[str, object]] = {}

        best_overall_return = -float("inf")
        best_action = 0

        # Lưu vết trạng thái hiện tại vào lịch sử gần
        self.recent_visited_states.append(current_state)
        if len(self.recent_visited_states) > 10:
            self.recent_visited_states.pop(0)

        # Duyệt qua từng hành động đầu tiên có thể chọn
        for first_action in range(num_actions):
            best_branch_return = -float("inf")
            best_branch_path: List[int] = []

            # Sinh tất cả chuỗi hành động tương lai: horizon - 1
            remaining_depth = self.planning_horizon - 1
            future_sub_sequences = list(
                itertools.product(range(num_actions), repeat=remaining_depth)
            )

            for sub_seq in future_sub_sequences:
                full_seq = (first_action,) + sub_seq
                ret, path = self.simulate_trajectory(
                    current_state, full_seq, recent_history=self.recent_visited_states
                )
                if ret > best_branch_return:
                    best_branch_return = ret
                    best_branch_path = path

            branch_analysis[first_action] = {
                "expected_return": best_branch_return,
                "best_imagined_path": best_branch_path,
            }

            if best_branch_return > best_overall_return:
                best_overall_return = best_branch_return
                best_action = first_action

        return best_action, branch_analysis