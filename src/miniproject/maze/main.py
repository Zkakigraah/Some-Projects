import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from miniproject.maze.grid import Maze, CharMapping
from miniproject.maze.solvers import (
    SearchResult,
    BaseSolver,
    BFSSolver,
    DFSSolver,
    DijkstraSolver,
    AStarSolver,
    BidirectionalAStarSolver,
    JPSSolver,
    NeuralPolicySolver,
)
from miniproject.maze.visualizer import MazeVisualizer, TerminalColors


def run_fog_benchmark(maze: Maze, maze_title: str, radius: int = 1, animate: bool = False) -> None:
    """Benchmark Level 3: So sánh điều kiện Toàn tri (God View) vs Quan sát một phần (Fog of War)."""
    from miniproject.maze.fog import FogOfWarEnvironment
    from miniproject.maze.online_solvers import (
        OptimisticReplanningAStar,
        OnlineSensorTremauxSolver,
        OnlineSearchResult,
    )

    print(f"\n{'='*75}")
    print(f"{TerminalColors.BOLD}{TerminalColors.CYAN}BẮT ĐẦU BENCHMARK LEVEL 3: FOG OF WAR (QUAN SÁT MỘT PHẦN) - {maze_title}{TerminalColors.RESET}")
    print(f"{'='*75}")
    print(f"Kích thước mê cung: {maze.height}x{maze.width} | Tầm nhìn cảm biến bán kính R={radius} (Cửa sổ {2*radius+1}x{2*radius+1})")
    print(f"Vị trí bắt đầu: {maze.start} | Vị trí đích: {maze.goal}\n")

    # 1. Chạy Oracle (A* Offline có toàn bộ bản đồ trước) làm mốc tham chiếu
    oracle_solver = AStarSolver()
    oracle_res = oracle_solver.solve(maze)
    oracle_cost = oracle_res.total_cost if oracle_res.success else float("inf")

    print(f"-> Đường đi tối ưu lý thuyết (Toàn tri - Oracle A*): {oracle_cost:.0f} bước | Thời gian: {oracle_res.runtime_ms:.2f} ms\n")

    # 2. Khởi tạo môi trường Sương mù
    env = FogOfWarEnvironment(maze, fov_radius=radius, known_goal=True)

    online_solvers = [
        OptimisticReplanningAStar(),
        OnlineSensorTremauxSolver(),
    ]

    results: List[OnlineSearchResult] = []
    for solver in online_solvers:
        print(f"-> Đang thực thi {solver.name} trong màn sương mù...")
        res = solver.solve(env, max_steps=max(5000, maze.height * maze.width * 3), animate=animate)
        results.append(res)

    # In bảng so sánh
    header = f"{'Thuật toán Online (Fog of War)':<42} | {'Số bước đi':<11} | {'Lệch Oracle':<12} | {'Số lần Replan':<13} | {'Độ phủ (%)':<10} | {'Thời gian (ms)':<14}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(f"{TerminalColors.BOLD}{header}{TerminalColors.RESET}")
    print(sep)

    for r in results:
        if not r.success:
            print(f"{r.algorithm_name:<42} | {'THẤT BẠI':<11} | {'-':<12} | {r.replanning_calls:<13} | {r.coverage_percent:<10.1f} | {r.runtime_ms:<14.2f}")
            continue

        diff_str = f"+{r.total_steps - oracle_cost:.0f} bước" if r.total_steps > oracle_cost else "Tối ưu"
        opt_color = TerminalColors.GREEN if r.total_steps == oracle_cost else TerminalColors.YELLOW

        print(
            f"{r.algorithm_name:<42} | "
            f"{r.total_steps:<11} | "
            f"{opt_color}{diff_str:<12}{TerminalColors.RESET} | "
            f"{r.replanning_calls:<13} | "
            f"{r.coverage_percent:<10.1f}% | "
            f"{r.runtime_ms:<14.2f}"
        )
    print(sep + "\n")

    print(f"{TerminalColors.BOLD}💡 BẢN ĐỒ KHÁM PHÁ CUỐI CÙNG (Sau khi Agent hoàn thành nhiệm vụ):{TerminalColors.RESET}")
    print(env.render_ansi(show_trail=True))


def analyze_benchmark(results: List[Tuple[str, SearchResult]]) -> None:
    successful = [res for _, res in results if res.success]
    if not successful:
        print(f"{TerminalColors.RED}Tất cả thuật toán đều không tìm thấy đường đi!{TerminalColors.RESET}")
        return

    min_nodes = min(res.nodes_explored_count for res in successful)
    min_cost = min(res.total_cost for res in successful)

    print(f"\n{TerminalColors.BOLD}💡 PHÂN TÍCH SO SÁNH:{TerminalColors.RESET}")
    for name, res in results:
        if not res.success:
            print(f"- {name}: Thất bại (ngõ cụt hoặc không tồn tại đường đi).")
            continue

        traits = []
        if abs(res.total_cost - min_cost) < 1e-6:
            traits.append(f"{TerminalColors.GREEN}Đạt đường đi tối ưu{TerminalColors.RESET}")
        else:
            traits.append(f"{TerminalColors.YELLOW}Đường đi dài hơn ({res.total_cost - min_cost:+.0f} bước){TerminalColors.RESET}")

        if res.nodes_explored_count == min_nodes:
            traits.append(f"{TerminalColors.CYAN}Khám phá ít node nhất ({res.nodes_explored_count} ô){TerminalColors.RESET}")

        print(f"- {TerminalColors.BOLD}{name}{TerminalColors.RESET}: {', '.join(traits)} | Tốc độ: {res.runtime_ms:.4f} ms")


def run_benchmark(
    maze: Maze,
    maze_title: str,
    model_path: Optional[str] = None,
    show_map: bool = False
) -> List[Tuple[str, SearchResult]]:
    print(f"\n{'='*70}")
    print(f"{TerminalColors.BOLD}{TerminalColors.CYAN}BẮT ĐẦU BENCHMARK: {maze_title}{TerminalColors.RESET}")
    print(f"{'='*70}")

    print(f"Kích thước mê cung: {maze.height}x{maze.width}")
    print(f"Vị trí bắt đầu (A/S): {maze.start} | Vị trí đích (G/E): {maze.goal}\n")

    solvers: List[BaseSolver] = [
        BFSSolver(),
        DFSSolver(),
        DijkstraSolver(),
        AStarSolver(),
        BidirectionalAStarSolver(),
        JPSSolver(),
    ]

    default_model = Path(__file__).resolve().parent / "maze_policy.pth"
    actual_model = model_path or (str(default_model) if default_model.exists() else None)
    if actual_model and Path(actual_model).exists():
        solvers.append(NeuralPolicySolver(model_path=actual_model))

    results: List[Tuple[str, SearchResult]] = []
    for solver in solvers:
        print(f"-> Đang thực thi {solver.name}...")
        res = solver.solve(maze)
        results.append((solver.name, res))

    is_large = maze.height > 35 or maze.width > 60
    if not is_large or show_map:
        print("\n" + f"{TerminalColors.BOLD}--- BẢN ĐỒ KẾT QUẢ ĐƯỜNG ĐI (Kèm các ô đã duyệt '·') ---{TerminalColors.RESET}")
        for name, res in results:
            print(f"\n{TerminalColors.BOLD}[{name}]{TerminalColors.RESET}")
            print(MazeVisualizer.render_solved_maze(maze, res, show_explored=True))
    else:
        print(f"\n{TerminalColors.DIM}(Mê cung kích thước lớn {maze.height}x{maze.width}: Đã ẩn bản đồ text để tránh vỡ màn hình Terminal. Dùng cờ --show-map để hiển thị nếu cần){TerminalColors.RESET}")

    MazeVisualizer.print_comparison_table(results)
    analyze_benchmark(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Maze Solver Benchmark & Inspector")
    parser.add_argument("--file", "-f", type=str, help="Đường dẫn file text chứa maze.")
    parser.add_argument("--image", "--pdf", "-i", type=str, help="Đường dẫn file ảnh (.png, .jpg) hoặc file PDF (.pdf).")
    parser.add_argument("--start", "-s", type=str, help="Tùy chọn tọa độ Start dạng 'row,col'.")
    parser.add_argument("--goal", "-g", type=str, help="Tùy chọn tọa độ Goal dạng 'row,col'.")
    parser.add_argument("--grid-size", type=str, default=None, help="Tùy chọn kích thước lưới ô 'rows,cols' (ví dụ: '25,40' hoặc '30,40').")
    parser.add_argument("--model", "-m", type=str, default=None, help="Đường dẫn checkpoint mạng nơ-ron (.pth).")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Tên file hoặc đường dẫn lưu ảnh giải pháp (mặc định tự động lưu vào 'src/miniproject/maze/solved/')."
    )
    parser.add_argument("--show-map", action="store_true", help="Bắt buộc in toàn bộ bản đồ text trên terminal.")
    # Các cờ mở rộng Level 3 (Fog of War)
    parser.add_argument("--fog", action="store_true", help="Bật chế độ Level 3: Partial Observability (Fog of War / Sương mù).")
    parser.add_argument("--fog-radius", "-r", type=int, default=1, help="Bán kính cảm biến tầm nhìn của Agent (mặc định R=1 tức cửa sổ 3x3).")
    parser.add_argument("--animate", action="store_true", help="Hiển thị chuyển động hoạt họa tan sương mù trong thời gian thực.")
    args = parser.parse_args()

    custom_start = tuple(map(int, args.start.split(","))) if args.start else None
    custom_goal = tuple(map(int, args.goal.split(","))) if args.goal else None
    grid_size = tuple(map(int, args.grid_size.split(","))) if args.grid_size else None

    # Tải maze từ các nguồn
    if args.image:
        from miniproject.maze.vision import MazeVisionParser
        print(f"Đang xử lý thị giác (Computer Vision) từ file: {args.image}")
        maze, crop_box = MazeVisionParser.from_image_or_pdf(args.image, target_grid_size=grid_size)
        if custom_start:
            maze.start = custom_start
        if custom_goal:
            maze.goal = custom_goal
        title = f"Vision Input ({Path(args.image).name})"
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"{TerminalColors.RED}Lỗi: Không tìm thấy file '{args.file}'!{TerminalColors.RESET}")
            sys.exit(1)
        print(f"Đang đọc maze từ file: {file_path.resolve()}")
        maze = Maze.from_file(str(file_path), custom_start=custom_start, custom_goal=custom_goal)
        title = f"Custom File ({file_path.name})"
    else:
        sample_maze = """
##########
#A       #
# ### ## #
#   #    #
###   ## #
#      G #
##########
"""
        maze = Maze(sample_maze)
        title = "Maze nhỏ mẫu"

    # Chế độ chạy: Level 3 (Fog of War) hay Level 1/2 Benchmark chuẩn
    if args.fog:
        run_fog_benchmark(maze, title, radius=args.fog_radius, animate=args.animate)
    else:
        results = run_benchmark(maze, title, model_path=args.model, show_map=args.show_map)
        if args.image:
            optimal_res = next((res for name, res in results if "A*" in name and res.success), None)
            if optimal_res and optimal_res.path:
                from miniproject.maze.vision import MazeVisionParser

                # Xác định thư mục solved chuẩn bên trong package maze: src/miniproject/maze/solved/
                maze_package_dir = Path(__file__).resolve().parent
                solved_dir = maze_package_dir / "solved"
                solved_dir.mkdir(parents=True, exist_ok=True)

                if args.output:
                    user_out = Path(args.output)
                    # Nếu người dùng chỉ truyền tên file (ví dụ: -o solution.png hoặc my_sol.png)
                    if not user_out.is_absolute() and len(user_out.parts) == 1:
                        final_output_path = solved_dir / user_out
                    else:
                        final_output_path = user_out
                else:
                    # Mặc định tự động đặt tên theo file ảnh đầu vào (ví dụ: maze4.png -> maze4_sol.png)
                    input_stem = Path(args.image).stem
                    final_output_path = solved_dir / f"{input_stem}_sol.png"

                MazeVisionParser.draw_solution_on_image(
                    input_image_path=args.image,
                    path=optimal_res.path,
                    grid_shape=(maze.height, maze.width),
                    crop_box=crop_box,
                    output_image_path=str(final_output_path)
                )
                print(f"\n{TerminalColors.GREEN}✓ Đã vẽ đường đi tối ưu lên ảnh và lưu tại: {final_output_path.resolve()}{TerminalColors.RESET}")


if __name__ == "__main__":
    main()
    
# uv run .\src\miniproject\maze\main.py --file src\miniproject\maze\data\my_maze.txt <--grid-size xx,xx if possible>
# uv run .\src\miniproject\maze\main.py --image src\miniproject\maze\data\my_maze.pdf , .png, .img, .jpeg..,