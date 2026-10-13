import io
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from miniproject.maze.grid import Maze, Coordinate, CellType
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
from miniproject.maze.vision import MazeVisionParser
from miniproject.maze.fog import FogOfWarEnvironment, FogCellState
from miniproject.maze.online_solvers import (
    OptimisticReplanningAStar,
    OnlineSensorTremauxSolver,
    OnlineSearchResult,
)

# Cấu hình giao diện Streamlit toàn màn hình với theme công nghệ hiện đại
st.set_page_config(
    page_title="Maze AI & Algorithm Studio",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Cố định padding đỉnh trang tránh bị thanh điều hướng Streamlit che khuất tiêu đề */
    .block-container {
        padding-top: 4.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
        line-height: 1.25;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 2px;
    }
    .algo-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class AlgorithmAnimationEngine:
    """
    Bộ động cơ sinh chuỗi khung hình động (Animation Frame Generator) riêng biệt
    cho từng thuật toán, thể hiện trực quan cơ chế hoạt động thực tế.
    """

    @staticmethod
    def sample_sequence(full_list: List[Any], max_frames: int = 400) -> List[Any]:
        """Lấy mẫu khung hình mịn để vừa đảm bảo chuyển động chậm rãi, vừa mượt mà."""
        if len(full_list) <= max_frames:
            return full_list
        step = len(full_list) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        if indices[-1] != len(full_list) - 1:
            indices.append(len(full_list) - 1)
        return [full_list[i] for i in indices]

    @classmethod
    def generate_bfs_animation(cls, maze: Maze, res: SearchResult) -> Dict:
        """BFS Animation: Lan truyền dạng sóng vàng hoàng kim phủ kín hành lang."""
        explored = cls.sample_sequence(res.explored_order, max_frames=350)
        return {
            "type": "bfs",
            "name": "BFS: Lan tỏa sóng nước hoàng kim",
            "explored_frames": explored,
            "path": res.path,
            "theme": {"explored_color": "#facc15", "head_color": "#ffffff", "path_color": "#ef4444"},
        }

    @classmethod
    def generate_dfs_animation(cls, maze: Maze, res: SearchResult) -> Dict:
        """DFS Animation: Nhánh thăm dò sâu với vệt vàng đậm uốn lượn."""
        explored = cls.sample_sequence(res.explored_order, max_frames=450)
        return {
            "type": "dfs",
            "name": "DFS: Thăm dò nhánh sâu",
            "explored_frames": explored,
            "path": res.path,
            "theme": {"explored_color": "#eab308", "head_color": "#fef08a", "path_color": "#ef4444"},
        }

    @classmethod
    def generate_astar_animation(cls, maze: Maze, res: SearchResult) -> Dict:
        """A* Animation: Luồng quét định hướng Heuristic màu vàng rực rỡ."""
        explored = cls.sample_sequence(res.explored_order, max_frames=350)
        return {
            "type": "astar",
            "name": "A* Search: Quét định hướng Heuristic",
            "explored_frames": explored,
            "path": res.path,
            "theme": {"explored_color": "#facc15", "head_color": "#ffffff", "path_color": "#ef4444"},
        }

    @classmethod
    def generate_bidirectional_animation(cls, maze: Maze, res: SearchResult) -> Dict:
        """
        Bidirectional A*: Hai đầu đồng thời xuất phát:
        - Luồng 1 từ START lan tỏa (Vàng rực rỡ #facc15)
        - Luồng 2 từ GOAL lan tỏa (Hồng tím neon #ec4899 hoặc Cam rực #f97316)
        - Hai luồng chuyển động song song, gặp nhau ở giữa rồi bừng sáng đường đi ĐỎ nổi bật!
        """
        meta = getattr(res, "metadata", {}) or {}
        fwd_raw = meta.get("fwd_explored", [])
        bwd_raw = meta.get("bwd_explored", [])

        # Cơ chế dự phòng thông minh nếu metadata trống
        if not fwd_raw or not bwd_raw:
            fwd_raw, bwd_raw = [], []
            start, goal = maze.start, maze.goal
            for cell in res.explored_order:
                d_start = abs(cell[0] - start[0]) + abs(cell[1] - start[1])
                d_goal = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
                if d_start <= d_goal:
                    fwd_raw.append(cell)
                else:
                    bwd_raw.append(cell)

        # Cân đối khung hình cho 2 đầu chạy song song đồng bộ
        max_f = 350
        fwd_frames = cls.sample_sequence(fwd_raw, max_frames=max_f)
        bwd_frames = cls.sample_sequence(bwd_raw, max_frames=max_f)

        return {
            "type": "bidirectional",
            "name": "Bidirectional A*: Hai đầu cùng lúc hội tụ",
            "explored_fwd": fwd_frames,
            "explored_bwd": bwd_frames,
            "path": res.path,
            "theme": {
                "fwd_color": "#facc15",       # Vàng hoàng kim từ Start
                "bwd_color": "#ec4899",       # Hồng tím neon từ Goal
                "head_fwd": "#ffffff",
                "head_bwd": "#f472b6",
                "path_color": "#ef4444",      # Đỏ đậm nổi bật khi nối đích
            },
        }

    @classmethod
    def generate_jps_animation(cls, maze: Maze, res: SearchResult) -> Dict:
        """Jump Point Search: Quét tia nhảy cóc qua các hành lang."""
        explored = cls.sample_sequence(res.explored_order, max_frames=300)
        return {
            "type": "jps",
            "name": "JPS: Nhảy cóc hành lang",
            "explored_frames": explored,
            "path": res.path,
            "theme": {"explored_color": "#facc15", "head_color": "#ffffff", "path_color": "#ef4444"},
        }

    @classmethod
    def generate_neural_animation(cls, maze: Maze, res: SearchResult) -> Dict:
        """Neural Policy: Hộp quét cục bộ 15x15 + định hướng bước đi."""
        steps = cls.sample_sequence(res.explored_order, max_frames=400)
        return {
            "type": "neural",
            "name": "Neural Policy: Cửa sổ cục bộ 15x15",
            "agent_steps": steps,
            "path": res.path,
            "theme": {"explored_color": "#facc15", "head_color": "#ffffff", "path_color": "#ef4444"},
        }

    @classmethod
    def generate_fog_animation(cls, env: FogOfWarEnvironment, online_res: OnlineSearchResult) -> Dict:
        """Fog of War: Quầng sáng cảm biến tan sương mù."""
        path_taken = cls.sample_sequence(online_res.path_taken, max_frames=450)
        return {
            "type": "fog",
            "name": "Fog of War: Cảm biến vén màn sương",
            "path_taken": path_taken,
            "radius": env.fov_radius,
            "theme": {"fog_color": "#090d16", "sensor_color": "#38bdf8", "trail_color": "#facc15"},
        }

def render_interactive_canvas_player(
    maze: Maze,
    anim_data: Dict,
    height: int = 820,
    fps: int = 30
):
    """
    Trình phát Canvas HTML5:
    - Thanh điều khiển và chú thích đặt ở phía trên, siêu gọn gàng.
    - Sửa triệt để lỗi cắt xén mép trên nhờ safe center + margin: auto.
    - Mở rộng vùng hiển thị canvas lên đến 700px chiều cao.
    """
    grid_json = json.dumps(maze.grid)
    start_json = json.dumps(maze.start)
    goal_json = json.dumps(maze.goal)
    anim_json = json.dumps(anim_data)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 4px;
                background-color: #0b1120;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                user-select: none;
                overflow: hidden;
            }}
            .controls-panel {{
                display: flex;
                flex-direction: column;
                gap: 6px;
                margin-bottom: 8px;
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 8px 14px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }}
            .controls {{
                display: flex;
                align-items: center;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .btn {{
                background: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 12px;
                font-size: 0.85rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .btn:hover {{
                background: #facc15;
                color: #0f172a;
                border-color: #facc15;
            }}
            .slider {{
                flex-grow: 1;
                accent-color: #facc15;
                cursor: pointer;
                min-width: 140px;
            }}
            .speed-select {{
                background: #1e293b;
                color: #facc15;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 0.8rem;
                font-weight: 600;
            }}
            .status-badge {{
                font-size: 0.85rem;
                font-weight: bold;
                color: #facc15;
                min-width: 130px;
            }}
            .legend {{
                display: flex;
                gap: 14px;
                font-size: 0.78rem;
                color: #94a3b8;
                align-items: center;
                padding-top: 5px;
                border-top: 1px solid #1e293b;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .color-box {{
                width: 10px;
                height: 10px;
                border-radius: 2px;
            }}
            #canvas-container {{
                display: flex;
                justify-content: safe center;
                align-items: safe center;
                background: #030712;
                border: 1px solid #1f2937;
                border-radius: 10px;
                overflow: auto;
                width: 100%;
                box-sizing: border-box;
                max-height: 700px;
                min-height: 480px;
                box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6);
                position: relative;
                padding: 6px;
            }}
            canvas {{
                display: block;
                margin: auto;
                image-rendering: pixelated;
                image-rendering: crisp-edges;
            }}
        </style>
    </head>
    <body>
        <div class="controls-panel">
            <div class="controls-row">
                <button class="btn" id="playBtn">Tạm dừng ⏸</button>
                <button class="btn" id="restartBtn">Về đầu ⏮</button>
                <input type="range" id="timeSlider" class="slider" min="0" max="100" value="0">
                <span class="status-badge" id="stepCounter">Bước: 0 / 0</span>
                <select class="speed-select" id="speedSelect">
                    <option value="0.25">Tốc độ: 0.25x (Rất chậm)</option>
                    <option value="0.5" selected>Tốc độ: 0.5x (Chậm chuẩn)</option>
                    <option value="1">Tốc độ: 1.0x (Bình thường)</option>
                    <option value="2">Tốc độ: 2.0x (Nhanh)</option>
                    <option value="4">Tốc độ: 4.0x (Siêu nhanh)</option>
                </select>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="color-box" style="background:#22c55e;"></div><span>Điểm Start</span></div>
                <div class="legend-item"><div class="color-box" style="background:#facc15;"></div><span>Đang tìm kiếm (Vàng)</span></div>
                <div class="legend-item"><div class="color-box" style="background:#ec4899;"></div><span>Đầu Goal (Bidirectional)</span></div>
                <div class="legend-item"><div class="color-box" style="background:#ef4444; box-shadow: 0 0 6px #ef4444;"></div><span style="color:#ef4444; font-weight:bold;">Đường nghiệm (ĐỎ nổi bật)</span></div>
            </div>
        </div>
        <div id="canvas-container">
            <canvas id="mazeCanvas"></canvas>
        </div>

        <script>
            const grid = {grid_json};
            const start = {start_json};
            const goal = {goal_json};
            const anim = {anim_json};

            const rows = grid.length;
            const cols = grid[0].length;

            const canvas = document.getElementById('mazeCanvas');
            const ctx = canvas.getContext('2d');
            const playBtn = document.getElementById('playBtn');
            const restartBtn = document.getElementById('restartBtn');
            const timeSlider = document.getElementById('timeSlider');
            const stepCounter = document.getElementById('stepCounter');
            const speedSelect = document.getElementById('speedSelect');

            // Tính toán kích thước canvas tối đa được mở rộng
            const maxW = 1200;
            const maxH = 680;
            const cellSize = Math.max(2, Math.min(Math.floor(maxW / cols), Math.floor(maxH / rows)));
            canvas.width = cols * cellSize;
            canvas.height = rows * cellSize;

            let isPlaying = true;
            let currentStep = 0;
            let speedMultiplier = 0.5; // Mặc định chạy chậm rãi dễ nhìn
            let stepAccumulator = 0;
            let animationFrameId = null;

            // Xác định tổng số bước
            let totalSteps = 100;
            if (anim.type === 'fog') {{
                totalSteps = anim.path_taken.length;
            }} else if (anim.type === 'neural') {{
                totalSteps = anim.agent_steps.length;
            }} else if (anim.type === 'bidirectional') {{
                totalSteps = Math.max(anim.explored_fwd.length, anim.explored_bwd.length);
            }} else {{
                totalSteps = anim.explored_frames.length;
            }}
            timeSlider.max = totalSteps;

            function drawBaseGrid() {{
                ctx.fillStyle = '#030712';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                for (let r = 0; r < rows; r++) {{
                    for (let c = 0; c < cols; c++) {{
                        const cell = grid[r][c];
                        if (cell === '#') {{
                            ctx.fillStyle = '#1e293b';
                            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                        }} else if (cell === '~') {{
                            ctx.fillStyle = '#581c87';
                            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                        }}
                    }}
                }}
            }}

            function drawBoldSolutionPath(path) {{
                if (!path || path.length < 2) return;
                ctx.save();
                // Hiệu ứng bóng đỏ phát sáng nổi bật
                ctx.shadowColor = '#ef4444';
                ctx.shadowBlur = Math.max(8, cellSize * 1.2);
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = Math.max(4, cellSize * 0.85);
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                ctx.beginPath();
                path.forEach(([r, c], idx) => {{
                    const px = c * cellSize + cellSize / 2;
                    const py = r * cellSize + cellSize / 2;
                    if (idx === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }});
                ctx.stroke();

                // Nét lõi trắng nhẹ ở tâm đường đỏ giúp sắc nét tối đa
                ctx.shadowBlur = 0;
                ctx.strokeStyle = '#fca5a5';
                ctx.lineWidth = Math.max(1.5, cellSize * 0.3);
                ctx.stroke();
                ctx.restore();
            }}

            function renderFrame(step) {{
                drawBaseGrid();

                // 1. CHẾ ĐỘ BIDIRECTIONAL A* (Hai đầu đồng thời lan tỏa từ Start và Goal)
                if (anim.type === 'bidirectional') {{
                    const fwdTotal = anim.explored_fwd.length;
                    const bwdTotal = anim.explored_bwd.length;

                    // Tính tỉ lệ đồng bộ để 2 đầu chạy song song cùng lúc
                    const fwdCount = Math.min(fwdTotal, Math.floor((step / totalSteps) * fwdTotal));
                    const bwdCount = Math.min(bwdTotal, Math.floor((step / totalSteps) * bwdTotal));

                    const fwdSub = anim.explored_fwd.slice(0, fwdCount);
                    const bwdSub = anim.explored_bwd.slice(0, bwdCount);

                    // Luồng 1 từ START: Tô vệt VÀNG hoàng kim (#facc15)
                    ctx.fillStyle = 'rgba(250, 204, 21, 0.75)';
                    for (let i = 0; i < fwdSub.length; i++) {{
                        const [r, c] = fwdSub[i];
                        ctx.fillRect(c * cellSize + 0.5, r * cellSize + 0.5, cellSize - 1, cellSize - 1);
                    }}

                    // Luồng 2 từ GOAL: Tô vệt HỒNG TÍM neon (#ec4899)
                    ctx.fillStyle = 'rgba(236, 72, 153, 0.75)';
                    for (let i = 0; i < bwdSub.length; i++) {{
                        const [r, c] = bwdSub[i];
                        ctx.fillRect(c * cellSize + 0.5, r * cellSize + 0.5, cellSize - 1, cellSize - 1);
                    }}

                    // Đầu phát sóng START (Chấm sáng trắng vàng)
                    if (fwdSub.length > 0) {{
                        const [fr, fc] = fwdSub[fwdSub.length - 1];
                        ctx.fillStyle = '#ffffff';
                        ctx.fillRect(fc * cellSize, fr * cellSize, cellSize, cellSize);
                    }}

                    // Đầu phát sóng GOAL (Chấm sáng hồng neon)
                    if (bwdSub.length > 0) {{
                        const [br, bc] = bwdSub[bwdSub.length - 1];
                        ctx.fillStyle = '#fdf2f8';
                        ctx.fillRect(bc * cellSize, br * cellSize, cellSize, cellSize);
                    }}

                    // Khi 2 luồng chạm trán và kết thúc -> BẬT ĐƯỜNG ĐI ĐỎ NỔI BẬT!
                    if (step >= totalSteps && anim.path) {{
                        drawBoldSolutionPath(anim.path);
                    }}
                }}

                // 2. CHẾ ĐỘ FOG OF WAR
                else if (anim.type === 'fog') {{
                    ctx.fillStyle = 'rgba(3, 7, 18, 0.96)';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    const radius = anim.radius || 1;
                    const pathSub = anim.path_taken.slice(0, step + 1);

                    // Giải mã sương mù trong bán kính cảm biến
                    for (let i = 0; i < pathSub.length; i++) {{
                        const [ar, ac] = pathSub[i];
                        for (let dr = -radius; dr <= radius; dr++) {{
                            for (let dc = -radius; dc <= radius; dc++) {{
                                const nr = ar + dr;
                                const nc = ac + dc;
                                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {{
                                    ctx.fillStyle = (grid[nr][nc] === '#') ? '#1e293b' : '#0f172a';
                                    ctx.fillRect(nc * cellSize, nr * cellSize, cellSize, cellSize);
                                }}
                            }}
                        }}
                    }}

                    // Vết chân Agent (Vàng hoàng kim)
                    ctx.fillStyle = 'rgba(250, 204, 21, 0.65)';
                    for (let i = 0; i < pathSub.length - 1; i++) {{
                        const [pr, pc] = pathSub[i];
                        ctx.fillRect(pc * cellSize + cellSize*0.2, pr * cellSize + cellSize*0.2, cellSize*0.6, cellSize*0.6);
                    }}

                    // Đèn pin phát sáng Agent
                    if (pathSub.length > 0) {{
                        const [cr, cc] = pathSub[pathSub.length - 1];
                        ctx.fillStyle = '#facc15';
                        ctx.beginPath();
                        ctx.arc(cc * cellSize + cellSize/2, cr * cellSize + cellSize/2, Math.max(3, cellSize * 0.7), 0, Math.PI * 2);
                        ctx.fill();

                        ctx.strokeStyle = 'rgba(56, 189, 248, 0.8)';
                        ctx.lineWidth = 1.5;
                        ctx.strokeRect((cc - radius) * cellSize, (cr - radius) * cellSize, (2 * radius + 1) * cellSize, (2 * radius + 1) * cellSize);
                    }}
                }}

                // 3. CÁC THUẬT TOÁN ĐƠN ĐẦU (A*, BFS, DFS, JPS, Neural)
                else {{
                    const sub = (anim.type === 'neural')
                        ? anim.agent_steps.slice(0, step + 1)
                        : anim.explored_frames.slice(0, step + 1);

                    // Tô vệt VÀNG hoàng kim rõ nét (#facc15) cho tất cả ô đã duyệt
                    ctx.fillStyle = 'rgba(250, 204, 21, 0.75)';
                    for (let i = 0; i < sub.length; i++) {{
                        const [r, c] = sub[i];
                        ctx.fillRect(c * cellSize + 0.5, r * cellSize + 0.5, cellSize - 1, cellSize - 1);
                    }}

                    // Điểm quét hiện tại (Head) phát sáng trắng
                    if (sub.length > 0) {{
                        const [hr, hc] = sub[sub.length - 1];
                        ctx.fillStyle = '#ffffff';
                        ctx.fillRect(hc * cellSize, hr * cellSize, cellSize, cellSize);

                        if (anim.type === 'neural') {{
                            const rad = 7;
                            ctx.strokeStyle = 'rgba(234, 179, 8, 0.9)';
                            ctx.lineWidth = 1.5;
                            ctx.strokeRect((hc - rad) * cellSize, (hr - rad) * cellSize, (2 * rad + 1) * cellSize, (2 * rad + 1) * cellSize);
                        }}
                    }}

                    // Khi chạm đích -> ĐƯỜNG NGHIỆM ĐỎ NỔI BẬT ĐÈ LÊN MÀU VÀNG
                    if (step >= totalSteps && anim.path) {{
                        drawBoldSolutionPath(anim.path);
                    }}
                }}

                // Vẽ Start (Xanh lá to tròn) và Goal (Đỏ cờ)
                if (start) {{
                    ctx.fillStyle = '#22c55e';
                    ctx.beginPath();
                    ctx.arc(start[1] * cellSize + cellSize/2, start[0] * cellSize + cellSize/2, Math.max(4, cellSize * 0.9), 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }}
                if (goal) {{
                    ctx.fillStyle = '#ef4444';
                    ctx.beginPath();
                    ctx.arc(goal[1] * cellSize + cellSize/2, goal[0] * cellSize + cellSize/2, Math.max(4, cellSize * 0.9), 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }}

                stepCounter.innerText = `Bước: ${{step}} / ${{totalSteps}}`;
                timeSlider.value = step;
            }}

            function loop() {{
                if (isPlaying) {{
                    stepAccumulator += speedMultiplier;
                    if (stepAccumulator >= 1) {{
                        const advance = Math.floor(stepAccumulator);
                        stepAccumulator -= advance;
                        currentStep = Math.min(totalSteps, currentStep + advance);
                        renderFrame(currentStep);
                        if (currentStep >= totalSteps) {{
                            isPlaying = false;
                            playBtn.innerText = 'Phát lại 🔄';
                        }}
                    }}
                }}
                animationFrameId = requestAnimationFrame(loop);
            }}

            playBtn.onclick = () => {{
                if (currentStep >= totalSteps) {{
                    currentStep = 0;
                    isPlaying = true;
                    playBtn.innerText = 'Tạm dừng ⏸';
                }} else {{
                    isPlaying = !isPlaying;
                    playBtn.innerText = isPlaying ? 'Tạm dừng ⏸' : 'Tiếp tục ▶️';
                }}
            }};

            restartBtn.onclick = () => {{
                currentStep = 0;
                stepAccumulator = 0;
                isPlaying = false;
                playBtn.innerText = 'Tiếp tục ▶️';
                renderFrame(0);
            }};

            timeSlider.oninput = (e) => {{
                currentStep = parseInt(e.target.value);
                stepAccumulator = 0;
                renderFrame(currentStep);
            }};

            speedSelect.onchange = (e) => {{
                speedMultiplier = parseFloat(e.target.value);
                stepAccumulator = 0;
            }};

            // Khởi tạo
            renderFrame(0);
            loop();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)

def get_sample_maze_options() -> Dict[str, Any]:
    """Tải danh sách các file mê cung mẫu có sẵn trong thư mục dự án."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    samples = {}
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.*")):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf", ".txt"]:
                samples[f"📁 {p.name}"] = p

    # Mê cung mặc định nếu không có file data
    samples["✨ Mê cung mẫu 11x11"] = """
###########
#A        #
# ### ### #
#   #   # #
### # # # #
#   # # # #
# ### ### #
#         #
# ####### #
#        G#
###########
"""
    return samples

def main():
    st.markdown('<div class="main-header">🧭 Maze AI & Algorithm Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Phòng thí nghiệm tương tác trực quan hóa thuật toán tìm đường, Mạng Nơ-ron và Fog of War (POMDP)</div>',
        unsafe_allow_html=True,
    )

    sample_options = get_sample_maze_options()

    with st.sidebar:
        st.header("⚙️ Cấu hình Nguồn Mê cung")
        input_type = st.radio(
            "Chọn phương thức nạp bản đồ:",
            ["Tải lên file (Ảnh/PDF/Text)", "Chọn file mẫu trong Data", "Nhập trực tiếp ASCII"],
            index=1,
        )

        maze_obj: Optional[Maze] = None
        crop_box: Optional[Tuple[int, int, int, int]] = None
        input_image_path: Optional[str] = None
        maze_title: str = "Custom Maze"

        if input_type == "Tải lên file (Ảnh/PDF/Text)":
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc dán ảnh/file vào đây:",
                type=["png", "jpg", "jpeg", "pdf", "txt"],
                help="Hỗ trợ ảnh mê cung raster, file PDF, hoặc file text ASCII / Box-drawing.",
            )
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix.lower()
                temp_dir = Path(__file__).resolve().parent / "temp_uploads"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                maze_title = uploaded_file.name
                if suffix in [".png", ".jpg", ".jpeg", ".pdf"]:
                    input_image_path = str(temp_path)
                    maze_obj, crop_box = MazeVisionParser.from_image_or_pdf(str(temp_path))
                else:
                    maze_obj = Maze.from_file(str(temp_path))

        elif input_type == "Chọn file mẫu trong Data":
            selected_sample = st.selectbox("Danh sách mê cung có sẵn:", list(sample_options.keys()))
            target = sample_options[selected_sample]
            maze_title = selected_sample.replace("📁 ", "")

            if isinstance(target, Path):
                suffix = target.suffix.lower()
                if suffix in [".png", ".jpg", ".jpeg", ".pdf"]:
                    input_image_path = str(target)
                    maze_obj, crop_box = MazeVisionParser.from_image_or_pdf(str(target))
                else:
                    maze_obj = Maze.from_file(str(target))
            else:
                maze_obj = Maze(target)

        else:
            raw_text = st.text_area("Nhập ma trận văn bản:", value=sample_options["✨ Mê cung mẫu 11x11"], height=200)
            if raw_text.strip():
                maze_obj = Maze(raw_text)

        st.divider()
        st.header("🎛️ Chế độ Tìm kiếm")
        mode = st.radio(
            "Môi trường quan sát:",
            ["Toàn tri (God View - Classical & Neural)", "Sương mù (Fog of War - POMDP)"],
            index=0,
        )

        fog_radius = 1
        if mode == "Sương mù (Fog of War - POMDP)":
            fog_radius = st.slider("Bán kính cảm biến tầm nhìn (R):", min_value=1, max_value=4, value=1, help="R=1: cửa sổ 3x3, R=2: cửa sổ 5x5.")

    if maze_obj is None:
        st.info("👋 Vui lòng tải lên hoặc chọn một mê cung ở thanh bên trái để bắt đầu khám phá!")
        return

    # Hiển thị thông tin tổng quan của bản đồ
    col1, col2, col3, col4 = st.columns(4)
        # Hiển thị thông tin tổng quan của bản đồ
    col1, col2, col3, col4 = st.columns(4)
    with col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Kích thước Lưới</div><div class="metric-value">{maze_obj.height} × {maze_obj.width}</div></div>',
                unsafe_allow_html=True,
            )
    with col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Cổng Vào (Start)</div><div class="metric-value">{maze_obj.start}</div></div>',
                unsafe_allow_html=True,
            )
    with col3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Cổng Ra (Goal)</div><div class="metric-value">{maze_obj.goal}</div></div>',
                unsafe_allow_html=True,
            )
    with col4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Chế độ hiển thị</div><div class="metric-value">{"Sương mù" if "Fog" in mode else "Toàn tri"}</div></div>',
                unsafe_allow_html=True,
            )

    tab_anim, tab_benchmark, tab_image_solved = st.tabs(
        ["🎬 Trực quan hóa & Animation", "📊 Bảng Benchmark Chi tiết", "🖼️ Ảnh Nghiệm Solved"]
    )

    with tab_anim:
        if mode == "Toàn tri (God View - Classical & Neural)":
            algo_choice = st.selectbox(
                "Chọn Thuật toán để xem Animation đặc thù:",
                [
                    "A* Search (Manhattan Heuristic)",
                    "Jump Point Search (JPS - Nhảy cóc)",
                    "Bidirectional A* (Hai đầu hội tụ)",
                    "Breadth-First Search (BFS - Sóng nước)",
                    "Depth-First Search (DFS - Con rắn)",
                    "Dijkstra (Uniform Cost)",
                    "Learned Policy (Neural CNN)",
                ],
                index=0,
            )

            solver_map = {
                "A* Search (Manhattan Heuristic)": AStarSolver(),
                "Jump Point Search (JPS - Nhảy cóc)": JPSSolver(),
                "Bidirectional A* (Hai đầu hội tụ)": BidirectionalAStarSolver(),
                "Breadth-First Search (BFS - Sóng nước)": BFSSolver(),
                "Depth-First Search (DFS - Con rắn)": DFSSolver(),
                "Dijkstra (Uniform Cost)": DijkstraSolver(),
                "Learned Policy (Neural CNN)": NeuralPolicySolver(),
            }

            active_solver = solver_map[algo_choice]
            with st.spinner(f"Đang giải mê cung bằng {active_solver.name}..."):
                res = active_solver.solve(maze_obj)

            if not res.success:
                st.error("❌ Thuật toán không tìm thấy đường đi tới đích!")
            else:
                st.success(
                    f"✓ Hoàn thành: **{len(res.path)} bước đi** | Chi phí: **{res.total_cost:.1f}** | Khám phá: **{res.nodes_explored_count} ô** | Thời gian tính: **{res.runtime_ms:.2f} ms**"
                )

                # Sinh animation data tương ứng
                if "BFS" in algo_choice:
                    anim_data = AlgorithmAnimationEngine.generate_bfs_animation(maze_obj, res)
                elif "DFS" in algo_choice:
                    anim_data = AlgorithmAnimationEngine.generate_dfs_animation(maze_obj, res)
                elif "Bidirectional" in algo_choice:
                    anim_data = AlgorithmAnimationEngine.generate_bidirectional_animation(maze_obj, res)
                elif "JPS" in algo_choice:
                    anim_data = AlgorithmAnimationEngine.generate_jps_animation(maze_obj, res)
                elif "Neural" in algo_choice:
                    anim_data = AlgorithmAnimationEngine.generate_neural_animation(maze_obj, res)
                else:
                    anim_data = AlgorithmAnimationEngine.generate_astar_animation(maze_obj, res)

                render_interactive_canvas_player(maze_obj, anim_data, height=820)

        else:
            # Chế độ Fog of War
            online_algo_choice = st.selectbox(
                "Chọn Thuật toán Khám phá trong Màn sương:",
                ["Optimistic Online Replanning A* (Free-Space Assumption)", "Online Sensor Trémaux Exploration"],
            )

            env = FogOfWarEnvironment(maze_obj, fov_radius=fog_radius, known_goal=True)
            solver_instance = (
                OptimisticReplanningAStar()
                if "Optimistic" in online_algo_choice
                else OnlineSensorTremauxSolver()
            )

            with st.spinner("Đang cho Agent thám hiểm trong màn sương mù..."):
                online_res = solver_instance.solve(env, max_steps=max(4000, maze_obj.height * maze_obj.width * 2))

            if online_res.success:
                st.success(
                    f"✓ Agent đã đến đích an toàn! Tổng bước di chuyển: **{online_res.total_steps}** | Số lần Replan khi gặp tường: **{online_res.replanning_calls}** | Độ phủ sương mù đã tan: **{online_res.coverage_percent:.1f}%**"
                )
            else:
                st.warning("⚠️ Agent chưa đến được đích trong giới hạn bước cho phép.")

            fog_anim_data = AlgorithmAnimationEngine.generate_fog_animation(env, online_res)
            render_interactive_canvas_player(maze_obj, fog_anim_data, height=820)

    with tab_benchmark:
        st.subheader("🏁 So sánh Toàn bộ Thuật toán (God View)")
        benchmark_solvers: List[BaseSolver] = [
            BFSSolver(),
            DFSSolver(),
            DijkstraSolver(),
            AStarSolver(),
            BidirectionalAStarSolver(),
            JPSSolver(),
            NeuralPolicySolver(),
        ]

        if st.button("🚀 Chạy Benchmark Toàn bộ 7 Thuật toán"):
            benchmark_results = []
            progress_bar = st.progress(0)
            for idx, s in enumerate(benchmark_solvers):
                r = s.solve(maze_obj)
                benchmark_results.append({
                    "Thuật toán": s.name,
                    "Thành công": "✓" if r.success else "✗",
                    "Độ dài Path": len(r.path) if r.success else "-",
                    "Tổng Chi phí": f"{r.total_cost:.1f}" if r.success else "-",
                    "Số ô đã duyệt": r.nodes_explored_count,
                    "Thời gian (ms)": f"{r.runtime_ms:.2f}",
                })
                progress_bar.progress((idx + 1) / len(benchmark_solvers))

            st.dataframe(benchmark_results, use_container_width=True)

    with tab_image_solved:
        st.subheader("🖼️ Xuất và Lưu Ảnh Lời Giải")
        if input_image_path and crop_box:
            oracle_res = AStarSolver().solve(maze_obj)
            if oracle_res.success and oracle_res.path:
                solved_dir = Path(__file__).resolve().parent / "solved"
                solved_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(input_image_path).stem
                output_path = solved_dir / f"{stem}_sol.png"

                MazeVisionParser.draw_solution_on_image(
                    input_image_path=input_image_path,
                    path=oracle_res.path,
                    grid_shape=(maze_obj.height, maze_obj.width),
                    crop_box=crop_box,
                    output_image_path=str(output_path),
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Ảnh Gốc Đầu Vào:**")
                    st.image(input_image_path, use_container_width=True)
                with col_b:
                    st.markdown(f"**Ảnh Nghiệm Đã Vẽ Đường Đi ({output_path.name}):**")
                    st.image(str(output_path), use_container_width=True)

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Tải Ảnh Solved Về Máy",
                        data=f.read(),
                        file_name=output_path.name,
                        mime="image/png",
                    )
        else:
            st.info("💡 Tab này tự động hiển thị khi bạn tải lên một file Ảnh (.png, .jpg) hoặc file PDF.")

if __name__ == "__main__":
    main()
