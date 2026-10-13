import io
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image, ImageDraw
import pymupdf

from miniproject.maze.grid import Maze, CellType, Coordinate


class MazeVisionParser:
    """
    Bộ xử lý Thị giác máy tính (Computer Vision):
    - Đọc file ảnh (.png, .jpg) hoặc file PDF (.pdf).
    - Quét khe hở tự động trên TOÀN BỘ 4 CẠNH VIỀN (Trên, Dưới, Trái, Phải).
    - Tự động xác định cặp điểm vào/ra có khoảng cách xa nhất.
    - Bảo toàn 100% tính liên tục của tường đen (Morphological Barrier).
    - Vẽ đường nghiệm uốn lượn sắc nét, đậm rõ ràng theo tâm hành lang.
    """

    @staticmethod
    def pdf_to_image(pdf_path: str, page_index: int = 0) -> Image.Image:
        """Trích xuất trang PDF thành ảnh PIL bằng PyMuPDF."""
        doc = pymupdf.open(pdf_path)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=150)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    @staticmethod
    def _find_maze_bounding_box(gray_np: np.ndarray, threshold: int = 140) -> Tuple[int, int, int, int]:
        """Tìm 4 đường viền đen bao quanh của mê cung để loại bỏ lề giấy."""
        h_orig, w_orig = gray_np.shape
        is_dark = gray_np < threshold

        row_dens = np.mean(is_dark, axis=1)
        col_dens = np.mean(is_dark, axis=0)

        candidate_rows = np.where(row_dens > 0.15)[0]
        candidate_cols = np.where(col_dens > 0.15)[0]

        if len(candidate_rows) >= 2 and len(candidate_cols) >= 2:
            r_min, r_max = int(candidate_rows[0]), int(candidate_rows[-1])
            c_min, c_max = int(candidate_cols[0]), int(candidate_cols[-1])
        else:
            r_min, r_max = 0, h_orig - 1
            c_min, c_max = 0, w_orig - 1

        return r_min, r_max, c_min, c_max

    @staticmethod
    def _find_white_runs(bool_1d: np.ndarray, min_len: int = 6, max_len: int = 120) -> List[Tuple[int, int]]:
        """Tìm tất cả các khoảng liên tục màu trắng trên một lát cắt viền."""
        runs: List[Tuple[int, int]] = []
        in_run = False
        start_idx = 0

        for i, val in enumerate(bool_1d):
            if val and not in_run:
                in_run = True
                start_idx = i
            elif not val and in_run:
                in_run = False
                length = i - start_idx
                if min_len <= length <= max_len:
                    runs.append((start_idx, i - 1))

        if in_run:
            length = len(bool_1d) - start_idx
            if min_len <= length <= max_len:
                runs.append((start_idx, len(bool_1d) - 1))

        return runs

    @classmethod
    def _detect_perimeter_openings(
        cls,
        cropped_np: np.ndarray,
        threshold: int = 140
    ) -> List[Dict]:
        """
        Quét độc lập cả 4 cạnh viền ngoài: Top, Bottom, Left, Right.
        Trả về danh sách tất cả các khe hở thực tế kèm tọa độ và hướng đi vào.
        """
        crop_h, crop_w = cropped_np.shape
        depth = max(2, min(5, crop_h // 20, crop_w // 20))
        openings: List[Dict] = []

        # 1. Cạnh Trên (Top)
        top_slice = cropped_np[0:depth, :]
        top_white = np.mean(top_slice, axis=0) > threshold
        for s, e in cls._find_white_runs(top_white):
            if 4 <= s and e <= crop_w - 5:
                openings.append({
                    "border": "top",
                    "pixel_r": 0,
                    "pixel_c": int(round((s + e) / 2.0)),
                    "width": e - s + 1,
                    "inward_dr": 1,
                    "inward_dc": 0,
                })

        # 2. Cạnh Dưới (Bottom)
        bottom_slice = cropped_np[max(0, crop_h - depth):crop_h, :]
        bottom_white = np.mean(bottom_slice, axis=0) > threshold
        for s, e in cls._find_white_runs(bottom_white):
            if 4 <= s and e <= crop_w - 5:
                openings.append({
                    "border": "bottom",
                    "pixel_r": crop_h - 1,
                    "pixel_c": int(round((s + e) / 2.0)),
                    "width": e - s + 1,
                    "inward_dr": -1,
                    "inward_dc": 0,
                })

        # 3. Cạnh Trái (Left)
        left_slice = cropped_np[:, 0:depth]
        left_white = np.mean(left_slice, axis=1) > threshold
        for s, e in cls._find_white_runs(left_white):
            if 4 <= s and e <= crop_h - 5:
                openings.append({
                    "border": "left",
                    "pixel_r": int(round((s + e) / 2.0)),
                    "pixel_c": 0,
                    "width": e - s + 1,
                    "inward_dr": 0,
                    "inward_dc": 1,
                })

        # 4. Cạnh Phải (Right)
        right_slice = cropped_np[:, max(0, crop_w - depth):crop_w]
        right_white = np.mean(right_slice, axis=1) > threshold
        for s, e in cls._find_white_runs(right_white):
            if 4 <= s and e <= crop_w - 5:
                openings.append({
                    "border": "right",
                    "pixel_r": int(round((s + e) / 2.0)),
                    "pixel_c": crop_w - 1,
                    "width": e - s + 1,
                    "inward_dr": 0,
                    "inward_dc": -1,
                })

        return openings

    @classmethod
    def from_image_or_pdf(
        cls,
        file_path: str,
        target_grid_size: Optional[Tuple[int, int]] = None,
        threshold: int = 140
    ) -> Tuple[Maze, Tuple[int, int, int, int]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        if path.suffix.lower() == ".pdf":
            pil_img = cls.pdf_to_image(str(path), page_index=0)
        else:
            pil_img = Image.open(str(path))

        pil_rgb = pil_img.convert("RGB")
        rgb_np = np.array(pil_rgb)
        h_orig, w_orig, _ = rgb_np.shape

        # 1. Phát hiện các nét vẽ đỏ (nhãn 'start', 'goal', 'S', 'G') nếu có
        is_red = (rgb_np[:, :, 0] > 140) & (rgb_np[:, :, 1] < 100) & (rgb_np[:, :, 2] < 100)
        red_coords = np.argwhere(is_red)

        detected_start_px: Optional[Tuple[int, int]] = None
        detected_goal_px: Optional[Tuple[int, int]] = None

        clean_rgb_np = rgb_np.copy()
        if len(red_coords) > 0:
            clean_rgb_np[is_red] = [255, 255, 255]
            mid_row = h_orig // 2
            top_reds = red_coords[red_coords[:, 0] < mid_row]
            bottom_reds = red_coords[red_coords[:, 0] >= mid_row]

            if len(top_reds) > 0:
                detected_start_px = tuple(np.mean(top_reds, axis=0).astype(int))
            if len(bottom_reds) > 0:
                detected_goal_px = tuple(np.mean(bottom_reds, axis=0).astype(int))

        gray_img = Image.fromarray(clean_rgb_np).convert("L")
        gray_np = np.array(gray_img)

        # 2. Định vị khung viền chữ nhật của mê cung
        r_min, r_max, c_min, c_max = cls._find_maze_bounding_box(gray_np, threshold=threshold)
        crop_box = (r_min, r_max, c_min, c_max)
        cropped_np = gray_np[r_min:r_max + 1, c_min:c_max + 1]
        crop_h, crop_w = cropped_np.shape

        # 3. Quét toàn bộ 4 cạnh viền ngoài tìm các khe hở
        openings = cls._detect_perimeter_openings(cropped_np, threshold=threshold)

        # Đo đạc kích thước hành lang từ các khe hở phát hiện được
        if openings:
            corridor_width = float(np.median([op["width"] for op in openings]))
            corridor_width = max(8.0, min(80.0, corridor_width))
        else:
            corridor_width = 22.0

        # Bước chia lưới stride sao cho 1 hành lang chứa ~5 ô, tường ~1 ô
        stride = max(2, int(round(corridor_width / 5.0)))
        grid_h = crop_h // stride
        grid_w = crop_w // stride

        # 4. Xây dựng ma trận nhị phân Morphological Barrier
        grid_matrix = [[CellType.EMPTY for _ in range(grid_w)] for _ in range(grid_h)]

        for r in range(grid_h):
            y1 = r * stride
            y2 = min(crop_h, (r + 1) * stride)
            for c in range(grid_w):
                x1 = c * stride
                x2 = min(crop_w, (c + 1) * stride)

                patch = cropped_np[y1:y2, x1:x2]
                if float(np.min(patch)) < threshold:
                    grid_matrix[r][c] = CellType.WALL

        start_coord: Coordinate = (0, 0)
        goal_coord: Coordinate = (grid_h - 1, grid_w - 1)

        # Trường hợp A: Có đánh dấu chữ đỏ
        if detected_start_px is not None and detected_goal_px is not None:
            sr = int(round((detected_start_px[0] - r_min) / stride))
            sc = int(round((detected_start_px[1] - c_min) / stride))
            gr = int(round((detected_goal_px[0] - r_min) / stride))
            gc = int(round((detected_goal_px[1] - c_min) / stride))

            start_coord = (max(0, min(grid_h - 1, sr)), max(0, min(grid_w - 1, sc)))
            goal_coord = (max(0, min(grid_h - 1, gr)), max(0, min(grid_w - 1, gc)))

            grid_matrix[start_coord[0]][start_coord[1]] = CellType.START
            grid_matrix[goal_coord[0]][goal_coord[1]] = CellType.GOAL

        # Trường hợp B: Tự động chọn cặp khe hở có khoảng cách Euclid xa nhất
        elif len(openings) >= 2:
            best_pair = (openings[0], openings[1])
            max_dist_sq = -1.0

            for i in range(len(openings)):
                for j in range(i + 1, len(openings)):
                    op1, op2 = openings[i], openings[j]
                    d_sq = (op1["pixel_r"] - op2["pixel_r"]) ** 2 + (op1["pixel_c"] - op2["pixel_c"]) ** 2
                    if d_sq > max_dist_sq:
                        max_dist_sq = d_sq
                        best_pair = (op1, op2)

            p1, p2 = best_pair
            if (p1["pixel_r"] + p1["pixel_c"]) <= (p2["pixel_r"] + p2["pixel_c"]):
                start_op, goal_op = p1, p2
            else:
                start_op, goal_op = p2, p1

            def apply_opening_to_grid(op: Dict, cell_type: str) -> Coordinate:
                sr = max(0, min(grid_h - 1, int(round(op["pixel_r"] / stride))))
                sc = max(0, min(grid_w - 1, int(round(op["pixel_c"] / stride))))
                grid_matrix[sr][sc] = cell_type

                # Mở thông thoáng các ô bước vào bên trong hành lang
                dr, dc = op["inward_dr"], op["inward_dc"]
                for step in range(1, 3):
                    nr = max(0, min(grid_h - 1, sr + dr * step))
                    nc = max(0, min(grid_w - 1, sc + dc * step))
                    grid_matrix[nr][nc] = CellType.EMPTY

                return (sr, sc)

            start_coord = apply_opening_to_grid(start_op, CellType.START)
            goal_coord = apply_opening_to_grid(goal_op, CellType.GOAL)

        else:
            # Fallback nếu viền kín
            start_coord = (1, 1)
            goal_coord = (grid_h - 2, grid_w - 2)
            grid_matrix[start_coord[0]][start_coord[1]] = CellType.START
            grid_matrix[goal_coord[0]][goal_coord[1]] = CellType.GOAL

        raw_layout = "\n".join("".join(row) for row in grid_matrix)
        maze = Maze(
            raw_layout=raw_layout,
            custom_start=start_coord,
            custom_goal=goal_coord
        )
        return maze, crop_box

    @staticmethod
    def draw_solution_on_image(
        input_image_path: str,
        path: List[Coordinate],
        grid_shape: Tuple[int, int],
        crop_box: Tuple[int, int, int, int],
        output_image_path: str = "solution.png",
        line_color: Tuple[int, int, int] = (235, 30, 30),
        line_width: Optional[int] = None
    ) -> None:
        """
        Vẽ đường đi tối ưu lên ảnh gốc:
        - Tăng kích thước nét vẽ đậm, dễ nhìn, nổi bật giữa các bức tường đen.
        - Thêm vòng tròn đánh dấu điểm Start (Xanh lá) và Goal (Xanh dương / Đỏ).
        """
        if Path(input_image_path).suffix.lower() == ".pdf":
            pil_img = MazeVisionParser.pdf_to_image(input_image_path, page_index=0)
        else:
            pil_img = Image.open(input_image_path).convert("RGB")

        draw = ImageDraw.Draw(pil_img)
        total_h, total_w = grid_shape
        r_min, r_max, c_min, c_max = crop_box
        crop_h = r_max - r_min + 1
        crop_w = c_max - c_min + 1

        stride_y = crop_h / total_h
        stride_x = crop_w / total_w

        # Chuyển đổi tọa độ ô sang pixel tâm hành lang
        pixel_points = []
        for R, C in path:
            py = r_min + int(round((R + 0.5) * stride_y))
            px = c_min + int(round((C + 0.5) * stride_x))
            pixel_points.append((px, py))

        if len(pixel_points) >= 2:
            # Độ dày nét vẽ đậm rõ ràng (mặc định ít nhất 5px hoặc 80% bề rộng hành lang)
            actual_width = line_width or max(5, int(round(min(stride_x, stride_y) * 0.85)))
            draw.line(pixel_points, fill=line_color, width=actual_width, joint="curve")

            # Vẽ điểm tròn đánh dấu START (Màu xanh lá rực rỡ)
            start_px = pixel_points[0]
            radius = max(6, actual_width + 3)
            draw.ellipse(
                [start_px[0] - radius, start_px[1] - radius, start_px[0] + radius, start_px[1] + radius],
                fill=(0, 220, 0),
                outline=(0, 100, 0),
                width=2
            )

            # Vẽ điểm tròn đánh dấu GOAL (Màu vàng cam nổi bật)
            goal_px = pixel_points[-1]
            draw.ellipse(
                [goal_px[0] - radius, goal_px[1] - radius, goal_px[0] + radius, goal_px[1] + radius],
                fill=(255, 140, 0),
                outline=(180, 50, 0),
                width=2
            )

        # Đảm bảo thư mục lưu tồn tại
        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(str(out_path))