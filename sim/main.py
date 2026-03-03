import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from util import get_joint_index, get_link_index
from robot import DifferentialRobot
from lidar import Lidar
from obstacle_avoidance import ObstacleAvoidance
from odometry import Odometry

# ===== 初期化 =====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("field.urdf", [0, 0, 0], useFixedBase=True)

yaw = math.pi / 2   # 90°
orn = p.getQuaternionFromEuler([0, 0, yaw])
robot_id = p.loadURDF("robot.urdf",[0.25, -0.25, 0.05],orn)

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

def angle_diff(a, b):
    """
    角度差を -π〜π に正規化
    """
    return np.arctan2(np.sin(a - b), np.cos(a - b))

# ===== 軌跡描画用の前ステップ位置 =====
_TRAJ_Z = 0.02  # 地面より少し上に描画
prev_odom_x, prev_odom_y   = init_pos[0], init_pos[1]
prev_true_x, prev_true_y   = init_pos[0], init_pos[1]

# 処理間引き用カウンタ
_step = 0
_SCAN_EVERY  = 6    # LiDARスキャン+回避計算: 240/6  = 40 Hz
_DRAW_EVERY  = 24   # 軌跡描画:              240/24 = 10 Hz
_PRINT_EVERY = 120  # コンソール出力:        240/120 =  2 Hz

# キャッシュ用初期値
_cached_distances = [1.0] * lidar.NUM_RAYS
_cached_avoid     = (0.0, 0.0)

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

    vx = 0
    vy = 0
    speed = 0.2

    if ord('i') in keys:
        vx = -speed
    if ord('k') in keys:
        vx = speed
    if ord('j') in keys:
        vy = -speed
    if ord('l') in keys:
        vy = speed

    # ===== 回避ベクトル =====
    cmd_x = vx + avoid_x
    cmd_y = vy + avoid_y

    norm = math.hypot(cmd_x, cmd_y)
    if norm > 1e-5:
        cmd_x /= norm
        cmd_y /= norm

    robot.set_velocity_vector(cmd_x, cmd_y)

    # ===== オドメトリ更新 =====
    odom.step()

    # ===== 軌跡描画・出力（間引き）=====
    if _step % _DRAW_EVERY == 0 or _step % _PRINT_EVERY == 0:
        odom_x, odom_y, odom_theta = odom.get_state()
        true_pos, true_orn = p.getBasePositionAndOrientation(robot_id)
        _, _, true_theta = p.getEulerFromQuaternion(true_orn)


    # ===== 軌跡描画・出力（間引き）=====
    if _step % _DRAW_EVERY == 0 or _step % _PRINT_EVERY == 0:
        odom_x, odom_y, odom_theta = odom.get_state()
        true_pos, true_orn = p.getBasePositionAndOrientation(robot_id)
        _, _, true_theta = p.getEulerFromQuaternion(true_orn)

        if _step % _DRAW_EVERY == 0:
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

        if _step % _PRINT_EVERY == 0:
            theta_err_odom = angle_diff(true_theta, odom_theta)
            print(
                f"X: (Tr {true_pos[0]:.2f}, Od {odom_x:.2f})  "
                f"Y: (Tr {true_pos[1]:.2f}, Od {odom_y:.2f})  "
                f"θ: (Tr {true_theta:.2f}, Od {odom_theta:.2f})  "
                f"θerr(O:{theta_err_odom:.3f})"
            )

    p.stepSimulation()
    time.sleep(dt)
