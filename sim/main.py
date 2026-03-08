import pybullet as p
import pybullet_data
import time
import math
import os
import numpy as np
import importlib.util
import matplotlib
matplotlib.use("TkAgg")   # PyBullet GUI と共存できる非ブロッキングバックエンド
import matplotlib.pyplot as plt

from util import get_joint_index, get_link_index
from robot import DifferentialRobot
from lidar import Lidar
from lidar_odm import LidarOdm
from ict import ICT
from obstacle_avoidance import ObstacleAvoidance
from odometry import Odometry

# ===== 初期化 =====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("field.urdf", [0, 0, 0], useFixedBase=True)

# ===== ロボットの初期姿勢を指定してurdfをロード =====
yaw = math.pi / 2   # 90°
orn = p.getQuaternionFromEuler([0, 0, yaw])
robot_id = p.loadURDF("robot.urdf",[0.25, -0.25, 0.05],orn)

# ===== 使用部品のインデックスを取得 =====
left = get_joint_index(robot_id, "left_wheel_joint")
right = get_joint_index(robot_id, "right_wheel_joint")
lidar_link = get_link_index(robot_id, "lidar_link")

robot = DifferentialRobot(robot_id, left, right)
lidar = Lidar(robot_id, lidar_link)

avoidance = ObstacleAvoidance(
    fov=lidar.FOV,
    num_rays=lidar.NUM_RAYS,
    safe_dist=0.15
)

init_pos, init_orn = p.getBasePositionAndOrientation(robot_id)
_, _, init_yaw = p.getEulerFromQuaternion(init_orn)

# 共通の物理パラメータ
wheel_radius = 0.03
wheel_base = 0.1
dt = 1./240.

# オドメトリのみ使用
odom = Odometry(
    robot_id,
    left,
    right,
    wheel_radius,
    wheel_base,
    dt,
    initial_x=init_pos[0],
    initial_y=init_pos[1],
    initial_theta=init_yaw
)

# ICT補正されない純粋なオドメトリ（比較表示専用・set_state()を呼ばない）
odom_pure = Odometry(
    robot_id,
    left,
    right,
    wheel_radius,
    wheel_base,
    dt,
    initial_x=init_pos[0],
    initial_y=init_pos[1],
    initial_theta=init_yaw
)

def angle_diff(a, b):
    """
    角度差を -π〜π に正規化
    """
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def _ndc_to_world(ndc_x: float, ndc_y: float, ndc_z: float = 0.0) -> list:
    """
    NDC座標 (ndc_x, ndc_y 共に -1～1) をワールド座標へ変換。
    カメラ姿勢に追従する画面固定位置の計算に使用。
    """
    cam = p.getDebugVisualizerCamera()
    V   = np.array(cam[2]).reshape(4, 4).T   # column-major → row-major
    P   = np.array(cam[3]).reshape(4, 4).T
    vp_inv = np.linalg.inv(P @ V)
    ndc_h  = np.array([ndc_x, ndc_y, ndc_z, 1.0])
    world_h = vp_inv @ ndc_h
    return (world_h[:3] / world_h[3]).tolist()

# ===== occupancy_grid_data 読み込み =====
_grid_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occupancy_grid_data.py")
_spec = importlib.util.spec_from_file_location("occupancy_grid_data", _grid_py)
_gmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gmod)
_grid      = np.array(_gmod.GRID, dtype=np.uint8)
_grid_res  = _gmod.RESOLUTION          # m/cell
_grid_orig = _gmod.ORIGIN              # (x_min, y_min) [m]

# ===== LidarOdm / ICT 初期化 =====
_lidar_odm = LidarOdm(
    _grid, _grid_res, _grid_orig,
    fov=lidar.FOV,
    num_rays=lidar.NUM_RAYS,
    max_dist=lidar.MAX_DIST,
)
_ict = ICT(_lidar_odm)

def _world_to_grid(wx: float, wy: float):
    """ワールド座標 → グリッドのピクセル座標 (col, row) に変換。"""
    col = int((wx - _grid_orig[0]) / _grid_res)
    row = int((wy - _grid_orig[1]) / _grid_res)
    return col, row

# ===== matplotlib マップウィンドウ初期化 =====
plt.ion()
_fig, _ax = plt.subplots(figsize=(5, 6))
_extent = [
    _grid_orig[0],
    _grid_orig[0] + _gmod.WIDTH  * _grid_res,
    _grid_orig[1],
    _grid_orig[1] + _gmod.HEIGHT * _grid_res,
]
_ax.imshow(_grid, origin="lower", cmap="gray_r", extent=_extent, vmin=0, vmax=1)
_ax.set_title("Occupancy Grid")
_ax.set_xlabel("X [m]")
_ax.set_ylabel("Y [m]")
_ax.set_aspect("equal")
# 現在位置マーカー（初期位置に配置）
_marker_true, = _ax.plot([], [], "go", markersize=8, label="True",  zorder=5)
_marker_odom, = _ax.plot([], [], "bo", markersize=8, label="Odom",  zorder=5)
_marker_ict,  = _ax.plot([], [], "ro", markersize=8, label="ICT",   zorder=5)
_ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.pause(0.001)
_TRAJ_Z = 0.02  # 地面より少し上に描画
prev_odom_x, prev_odom_y   = init_pos[0], init_pos[1]
prev_true_x, prev_true_y   = init_pos[0], init_pos[1]
prev_ict_x,  prev_ict_y    = init_pos[0], init_pos[1]
_ict_x, _ict_y, _ict_theta = init_pos[0], init_pos[1], init_yaw

# 処理間引き用カウンタ
_step = 0
_SCAN_EVERY  = 6    # LiDARスキャン+回避計算: 240/6  = 40 Hz
_DRAW_EVERY  = 24   # 軌跡描画:              240/24 = 10 Hz
_PRINT_EVERY = 120  # コンソール出力:        240/120 =  2 Hz
_ICT_EVERY   = 48   # ICTスキャンマッチング: 240/48 =  5 Hz

# キャッシュ用初期値
_cached_distances = [1.0] * lidar.NUM_RAYS
_cached_avoid     = (0.0, 0.0)

# ===== HUD テキスト（アニメーション内表示）用 ID =====
_hud_ids = [-1, -1, -1, -1]  # 4行分

# ===== 録画用 =====
_video_log_id  = -1   # -1 = 未録画
_video_dir     = os.path.dirname(os.path.abspath(__file__))
_video_counter = 0    # 連番（複数回録画対応）
_prev_r_key    = False  # チャタリング防止用

# ===== メインループ =====
while True:
    _step += 1

    # ===== LiDARスキャン・回避計算（間引き） =====
    if _step % _SCAN_EVERY == 0:
        _cached_distances, ray_from, ray_to, results = lidar.scan()
        _cached_avoid = avoidance.compute_avoid_vector(_cached_distances)
        lidar.draw(ray_from, ray_to, results)

    avoid_x, avoid_y = _cached_avoid

    # ===== キー入力 =====
    keys = p.getKeyboardEvents()

    # ===== 録画トグル (r キー) =====
    r_pressed = ord('r') in keys
    if r_pressed and not _prev_r_key:
        if _video_log_id == -1:
            _video_counter += 1
            _video_path = os.path.join(_video_dir, f"sim_record_{_video_counter:03d}.mp4")
            _video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, _video_path)
            print(f"[録画開始] {_video_path}")
        else:
            p.stopStateLogging(_video_log_id)
            _video_log_id = -1
            print(f"[録画停止] {_video_path} に保存しました")
    _prev_r_key = r_pressed

    # vx: 並進方向の入力（前進=+, 後退=-）  ※ワールドX軸ではない
    # vy: 旋回方向の入力（右旋回=+, 左旋回=-）※ワールドY軸への移動ではない
    vx = 0
    vy = 0
    speed = 0.2

    if ord('i') in keys:
        vx = -speed   # 後退
    if ord('k') in keys:
        vx = speed    # 前進
    if ord('j') in keys:
        vy = -speed   # 左旋回
    if ord('l') in keys:
        vy = speed    # 右旋回

    # ===== 回避ベクトル合成 =====
    # cmd_x → robot.set_velocity_vector 内で 並進速度 v [m/s] に変換
    # cmd_y → robot.set_velocity_vector 内で 角速度 omega [rad/s] に変換
    # （差動駆動のため横移動不可。cmd_y は旋回にマッピングされる）
    cmd_x = vx + avoid_x
    cmd_y = vy + avoid_y

    norm = math.hypot(cmd_x, cmd_y)
    if norm > 1e-5:
        cmd_x /= norm
        cmd_y /= norm

    robot.set_velocity_vector(cmd_x, cmd_y)

    # ===== オドメトリ更新 =====
    odom.step()
    odom_pure.step()  # ICT補正なし・表示専用

    # ===== ICT スキャンマッチング（間引き）=====
    if _step % _ICT_EVERY == 0:
        _base_x, _base_y, _base_theta = odom.get_state()
        _ict_x, _ict_y, _ict_theta, _ = _ict.match(
            _cached_distances, _base_x, _base_y, _base_theta
        )
        odom.set_state(_ict_x, _ict_y, _ict_theta)

    # ===== 軌跡描画・出力（間引き）=====
    if _step % _DRAW_EVERY == 0 or _step % _PRINT_EVERY == 0:
        # 表示には補正なし専用インスタンスを使用（ICTの影響ゼロ）
        odom_x, odom_y, odom_theta = odom_pure.get_state()
        true_pos, true_orn = p.getBasePositionAndOrientation(robot_id)
        _, _, true_theta = p.getEulerFromQuaternion(true_orn)

        if _step % _DRAW_EVERY == 0:
            # ===== グリッドマップ上の現在位置を更新 =====
            _marker_true.set_data([true_pos[0]], [true_pos[1]])
            _marker_odom.set_data([odom_x],      [odom_y])
            _marker_ict.set_data( [_ict_x],      [_ict_y])
            _fig.canvas.flush_events()

            # オドメトリ軌跡: 青
            if math.hypot(odom_x - prev_odom_x, odom_y - prev_odom_y) > 1e-4:
                p.addUserDebugLine(
                    [prev_odom_x, prev_odom_y, _TRAJ_Z],
                    [odom_x,      odom_y,      _TRAJ_Z],
                    lineColorRGB=[0.2, 0.4, 1.0],
                    lineWidth=2,
                )
                prev_odom_x, prev_odom_y = odom_x, odom_y

            # 真値軌跡: 緑
            tx, ty = true_pos[0], true_pos[1]
            if math.hypot(tx - prev_true_x, ty - prev_true_y) > 1e-4:
                p.addUserDebugLine(
                    [prev_true_x, prev_true_y, _TRAJ_Z],
                    [tx,          ty,          _TRAJ_Z],
                    lineColorRGB=[0.2, 0.9, 0.2],
                    lineWidth=2,
                )
                prev_true_x, prev_true_y = tx, ty

            # ICT補正軌跡: 赤
            if math.hypot(_ict_x - prev_ict_x, _ict_y - prev_ict_y) > 1e-4:
                p.addUserDebugLine(
                    [prev_ict_x, prev_ict_y, _TRAJ_Z + 0.001],
                    [_ict_x,     _ict_y,     _TRAJ_Z + 0.001],
                    lineColorRGB=[1.0, 0.2, 0.2],
                    lineWidth=2,
                )
                prev_ict_x, prev_ict_y = _ict_x, _ict_y

        if _step % _PRINT_EVERY == 0:
            theta_err_odom = angle_diff(true_theta, odom_theta)
            theta_err_ict  = angle_diff(true_theta, _ict_theta)
            hud_lines = [
                (f"True  X:{true_pos[0]:+.3f}  Y:{true_pos[1]:+.3f}  th:{math.degrees(true_theta):+.1f}deg", [0.2, 0.4, 1.0]),
                (f"Odom  X:{odom_x:+.3f}  Y:{odom_y:+.3f}  th:{math.degrees(odom_theta):+.1f}deg",           [0.2, 0.9, 0.2]),
                (f"ICT   X:{_ict_x:+.3f}  Y:{_ict_y:+.3f}  th:{math.degrees(_ict_theta):+.1f}deg",           [1.0, 0.2, 0.2]),
                (f"Err Odom:{math.degrees(theta_err_odom):+.1f}deg  ICT:{math.degrees(theta_err_ict):+.1f}deg", [0.0, 0.0, 0.0]),
            ]
            # addUserDebugText は \n 非対応のため1行ずつ描画
            # NDC y 座標をずらして縦に並べる (-0.74, -0.82, -0.90, -0.98)
            for i, ((text, color), ndc_y) in enumerate(zip(hud_lines, [-0.74, -0.82, -0.90, -0.98])):
                pos = _ndc_to_world(0.35, ndc_y)
                _hud_ids[i] = p.addUserDebugText(
                    text,
                    textPosition=pos,
                    textColorRGB=color,
                    textSize=1.0,
                    **({} if _hud_ids[i] == -1 else {"replaceItemUniqueId": _hud_ids[i]}),
                )

    p.stepSimulation()
    time.sleep(dt)
