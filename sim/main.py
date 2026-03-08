import pybullet as p
import pybullet_data
import time
import math
import os
import sys
import numpy as np
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

# ===========================================================================
# 移植区分の凡例
#   [ESP32移植対象]  実機 ESP32 上で C++ に移植して実行する処理
#   [SIM ONLY]       PyBullet シミュレーション専用。実機では不要または別手段で代替。
#
# ESP32移植ファイル対応表:
#   odometry.py  → Odometry クラス (エンコーダ積分)
#   lidar_odm.py → LidarOdm クラス (グリッドマップ上のレイマーチング)
#   ict.py       → ICT クラス      (スキャンマッチング・自己位置補正)
#   ※ lidar.py はシミュレーション用 LiDAR。実機では実 LiDAR ハードが距離配列を出力。
# ===========================================================================


# ---------------------------------------------------------------------------
# ユーティリティ  [ESP32移植対象]
# ---------------------------------------------------------------------------

def angle_diff(a: float, b: float) -> float:
    """角度差を -π〜π に正規化"""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


# ---------------------------------------------------------------------------
# ユーティリティ  [SIM ONLY]
# ---------------------------------------------------------------------------

def ndc_to_world(ndc_x: float, ndc_y: float, ndc_z: float = 0.0) -> list:
    """NDC座標 (-1〜1) をワールド座標へ変換（HUD位置計算用）"""
    cam = p.getDebugVisualizerCamera()
    V   = np.array(cam[2]).reshape(4, 4).T
    P   = np.array(cam[3]).reshape(4, 4).T
    vp_inv = np.linalg.inv(P @ V)
    ndc_h  = np.array([ndc_x, ndc_y, ndc_z, 1.0])
    world_h = vp_inv @ ndc_h
    return (world_h[:3] / world_h[3]).tolist()


# ---------------------------------------------------------------------------
# 初期化ヘルパー  [SIM ONLY]
# ---------------------------------------------------------------------------

def _init_simulation():
    """PyBullet・ロボット・センサ・推定器を初期化して返す。"""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.loadURDF("field.urdf", [0, 0, 0], useFixedBase=True)

    yaw = math.pi / 2
    orn = p.getQuaternionFromEuler([0, 0, yaw])
    robot_id = p.loadURDF("robot.urdf", [0.25, -0.25, 0.05], orn)

    left       = get_joint_index(robot_id, "left_wheel_joint")
    right      = get_joint_index(robot_id, "right_wheel_joint")
    lidar_link = get_link_index(robot_id,  "lidar_link")

    robot     = DifferentialRobot(robot_id, left, right)
    lidar     = Lidar(robot_id, lidar_link)
    avoidance = ObstacleAvoidance(fov=lidar.FOV, num_rays=lidar.NUM_RAYS, safe_dist=0.15)

    init_pos, init_orn = p.getBasePositionAndOrientation(robot_id)
    _, _, init_yaw = p.getEulerFromQuaternion(init_orn)

    wheel_radius = 0.03
    wheel_base   = 0.1
    dt           = 1. / 240.

    odom_kwargs = dict(
        robot_id=robot_id, left_joint=left, right_joint=right,
        wheel_radius=wheel_radius, wheel_base=wheel_base, dt=dt,
        initial_x=init_pos[0], initial_y=init_pos[1], initial_theta=init_yaw,
    )
    odom      = Odometry(**odom_kwargs)  # ICT補正あり（予測基点用）
    odom_pure = Odometry(**odom_kwargs)  # ICT補正なし（比較表示専用）

    return (robot_id, robot, lidar, avoidance,
            odom, odom_pure,
            wheel_radius, wheel_base, dt,
            init_pos, init_yaw)


def _init_map(lidar: Lidar):
    """occupancy_grid_data を読み込み LidarOdm / ICT を生成して返す。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import occupancy_grid_data as gmod

    grid      = np.array(gmod.GRID, dtype=np.uint8)
    grid_res  = gmod.RESOLUTION
    grid_orig = gmod.ORIGIN

    lidar_odm = LidarOdm(
        grid, grid_res, grid_orig,
        fov=lidar.FOV, num_rays=lidar.NUM_RAYS, max_dist=lidar.MAX_DIST,
    )
    ict = ICT(lidar_odm)

    return grid, grid_res, grid_orig, gmod.WIDTH, gmod.HEIGHT, ict


def _init_map_window(grid, grid_res, grid_orig, width, height):
    """matplotlib マップウィンドウを初期化して描画オブジェクトを返す。"""
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 6))
    extent = [
        grid_orig[0],
        grid_orig[0] + width  * grid_res,
        grid_orig[1],
        grid_orig[1] + height * grid_res,
    ]
    ax.imshow(grid, origin="lower", cmap="gray_r", extent=extent, vmin=0, vmax=1)
    ax.set_title("Occupancy Grid")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")

    marker_true, = ax.plot([], [], "go", markersize=8, label="True", zorder=5)
    marker_odom, = ax.plot([], [], "bo", markersize=8, label="Odom", zorder=5)
    marker_ict,  = ax.plot([], [], "ro", markersize=8, label="ICT",  zorder=5)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.pause(0.001)

    return fig, marker_true, marker_odom, marker_ict


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    # ===== 初期化 =====
    (robot_id, robot, lidar, avoidance,
     odom, odom_pure,
     wheel_radius, wheel_base, dt,
     init_pos, init_yaw) = _init_simulation()

    grid, grid_res, grid_orig, grid_w, grid_h, ict = _init_map(lidar)

    fig, marker_true, marker_odom, marker_ict = _init_map_window(
        grid, grid_res, grid_orig, grid_w, grid_h
    )

    # ===== 軌跡・HUD 用の状態変数 =====
    TRAJ_Z = 0.02
    prev_odom_x, prev_odom_y = init_pos[0], init_pos[1]
    prev_true_x, prev_true_y = init_pos[0], init_pos[1]
    prev_ict_x,  prev_ict_y  = init_pos[0], init_pos[1]
    ict_x, ict_y, ict_theta  = init_pos[0], init_pos[1], init_yaw
    hud_ids = [-1, -1, -1, -1]

    # ===== 処理間引き定数 (240Hz ベース) =====
    SCAN_EVERY  = 6    # 40 Hz
    DRAW_EVERY  = 24   # 10 Hz
    PRINT_EVERY = 120  #  2 Hz
    ICT_EVERY   = 48   #  5 Hz

    # ===== キャッシュ =====
    cached_distances = [1.0] * lidar.NUM_RAYS
    cached_avoid     = (0.0, 0.0)

    # ===== 録画用 =====
    video_log_id  = -1
    video_dir     = os.path.dirname(os.path.abspath(__file__))
    video_counter = 0
    prev_r_key    = False

    # ===== メインループ =====
    step = 0
    while True:
        step += 1

        # ----- [SIM ONLY] LiDARスキャン（PyBullet rayTestBatch で距離列を取得） -----
        # 実機では実 LiDAR ハードが cached_distances 相当の距離配列を直接出力する
        if step % SCAN_EVERY == 0:
            cached_distances, ray_from, ray_to, results = lidar.scan()
            cached_avoid = avoidance.compute_avoid_vector(cached_distances)
            lidar.draw(ray_from, ray_to, results)  # [SIM ONLY] レイ可視化

        avoid_x, avoid_y = cached_avoid

        # ----- [SIM ONLY] キー入力・録画・モータ指令 -----
        keys = p.getKeyboardEvents()

        # 録画トグル (r キー)
        r_pressed = ord('r') in keys
        if r_pressed and not prev_r_key:
            if video_log_id == -1:
                video_counter += 1
                video_path = os.path.join(video_dir, f"sim_record_{video_counter:03d}.mp4")
                video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
                print(f"[録画開始] {video_path}")
            else:
                p.stopStateLogging(video_log_id)
                video_log_id = -1
                print(f"[録画停止] {video_path} に保存しました")
        prev_r_key = r_pressed

        # 移動入力 (i=後退 / k=前進 / j=左旋回 / l=右旋回)
        speed = 0.2
        vx = -speed if ord('i') in keys else (speed if ord('k') in keys else 0.0)
        vy = -speed if ord('j') in keys else (speed if ord('l') in keys else 0.0)

        # 回避ベクトル合成・正規化
        cmd_x = vx + avoid_x
        cmd_y = vy + avoid_y
        norm  = math.hypot(cmd_x, cmd_y)
        if norm > 1e-5:
            cmd_x /= norm
            cmd_y /= norm
        robot.set_velocity_vector(cmd_x, cmd_y)  # [SIM ONLY] PyBullet モータ制御

        # ----- [ESP32移植対象] オドメトリ積分 (odometry.py → Odometry クラス) -----
        odom.step()
        odom_pure.step()  # [SIM ONLY] 比較表示専用（実機では不要）

        # ----- [ESP32移植対象] ICT スキャンマッチング・自己位置補正 -----
        #   使用クラス: lidar_odm.py (LidarOdm) + ict.py (ICT)
        #   cached_distances : 実機 LiDAR から取得した距離配列に相当
        if step % ICT_EVERY == 0:
            base_x, base_y, base_theta = odom.get_state()
            ict_x, ict_y, ict_theta, _ = ict.match(
                cached_distances, base_x, base_y, base_theta
            )
            odom.set_state(ict_x, ict_y, ict_theta)  # オドメトリ予測基点を補正

        # ----- [SIM ONLY] 描画・HUD 出力 -----
        if step % DRAW_EVERY == 0 or step % PRINT_EVERY == 0:
            # odom_pure は ICT の影響を受けない純粋な積分値（比較表示用）
            odom_x, odom_y, odom_theta = odom_pure.get_state()
            # odom は最後の ICT 補正点から毎ステップ積分済み → DRAW_EVERY 毎に新鮮な値
            # ICT 計算頻度（ICT_EVERY）に関わらず同じ 10Hz で更新される
            disp_ict_x, disp_ict_y, disp_ict_theta = odom.get_state()
            true_pos, true_orn = p.getBasePositionAndOrientation(robot_id)
            _, _, true_theta = p.getEulerFromQuaternion(true_orn)

            if step % DRAW_EVERY == 0:
                marker_true.set_data([true_pos[0]],  [true_pos[1]])
                marker_odom.set_data([odom_x],       [odom_y])
                marker_ict.set_data( [disp_ict_x],   [disp_ict_y])
                fig.canvas.flush_events()

                # オドメトリ軌跡（青）
                if math.hypot(odom_x - prev_odom_x, odom_y - prev_odom_y) > 1e-4:
                    p.addUserDebugLine(
                        [prev_odom_x, prev_odom_y, TRAJ_Z],
                        [odom_x,      odom_y,      TRAJ_Z],
                        lineColorRGB=[0.2, 0.4, 1.0], lineWidth=2,
                    )
                    prev_odom_x, prev_odom_y = odom_x, odom_y

                # 真値軌跡（緑）
                tx, ty = true_pos[0], true_pos[1]
                if math.hypot(tx - prev_true_x, ty - prev_true_y) > 1e-4:
                    p.addUserDebugLine(
                        [prev_true_x, prev_true_y, TRAJ_Z],
                        [tx,          ty,          TRAJ_Z],
                        lineColorRGB=[0.2, 0.9, 0.2], lineWidth=2,
                    )
                    prev_true_x, prev_true_y = tx, ty

                # ICT補正軌跡（赤）
                if math.hypot(disp_ict_x - prev_ict_x, disp_ict_y - prev_ict_y) > 1e-4:
                    p.addUserDebugLine(
                        [prev_ict_x,  prev_ict_y,  TRAJ_Z + 0.001],
                        [disp_ict_x,  disp_ict_y,  TRAJ_Z + 0.001],
                        lineColorRGB=[1.0, 0.2, 0.2], lineWidth=2,
                    )
                    prev_ict_x, prev_ict_y = disp_ict_x, disp_ict_y

            if step % PRINT_EVERY == 0:
                theta_err_odom = angle_diff(true_theta, odom_theta)
                theta_err_ict  = angle_diff(true_theta, disp_ict_theta)
                hud_lines = [
                    (f"True  X:{true_pos[0]:+.3f}  Y:{true_pos[1]:+.3f}  th:{math.degrees(true_theta):+.1f}deg", [0.2, 0.4, 1.0]),
                    (f"Odom  X:{odom_x:+.3f}  Y:{odom_y:+.3f}  th:{math.degrees(odom_theta):+.1f}deg",           [0.2, 0.9, 0.2]),
                    (f"ICT   X:{disp_ict_x:+.3f}  Y:{disp_ict_y:+.3f}  th:{math.degrees(disp_ict_theta):+.1f}deg", [1.0, 0.2, 0.2]),
                    (f"Err Odom:{math.degrees(theta_err_odom):+.1f}deg  ICT:{math.degrees(theta_err_ict):+.1f}deg", [0.0, 0.0, 0.0]),
                ]
                for i, ((text, color), ndc_y) in enumerate(zip(hud_lines, [-0.74, -0.82, -0.90, -0.98])):
                    pos = ndc_to_world(0.35, ndc_y)
                    hud_ids[i] = p.addUserDebugText(
                        text,
                        textPosition=pos,
                        textColorRGB=color,
                        textSize=1.0,
                        **({} if hud_ids[i] == -1 else {"replaceItemUniqueId": hud_ids[i]}),
                    )

        p.stepSimulation()  # [SIM ONLY]
        time.sleep(dt)


if __name__ == "__main__":
    main()
