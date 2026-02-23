import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from util import get_joint_index, get_link_index
from robot import DifferentialRobot
from lidar import Lidar
from obstacle_avoidance import ObstacleAvoidance
from complocalization import ComplementaryLocalization
from odometry import Odometry
from imu_localization import IMULocalization

# ===== 初期化 =====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("field.urdf", [0, 0, 0], useFixedBase=True)

yaw = math.pi / 2   # 90°
orn = p.getQuaternionFromEuler([0, 0, yaw])
robot_id = p.loadURDF("robot.urdf",[0.25, -0.25, 0.55],orn)

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
wheel_radius=0.03
wheel_base=0.1
dt=1./240.

# ３種類の推定方式の初期化
comp_loc = ComplementaryLocalization(
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

imu_loc = IMULocalization(
    robot_id=robot_id,
    dt=dt,
    accel_noise_std=0.05,
    gyro_noise_std=0.01,
    initial_x=init_pos[0],
    initial_y=init_pos[1],
    initial_theta=init_yaw
)

# ===== メインループ =====
while True:

    distances, ray_from, ray_to, results = lidar.scan()
    lidar.draw(ray_from, ray_to, results)

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
    avoid_x, avoid_y = avoidance.compute_avoid_vector(distances)

    cmd_x = vx + avoid_x
    cmd_y = vy + avoid_y

    norm = math.hypot(cmd_x, cmd_y)
    if norm > 1e-5:
        cmd_x /= norm
        cmd_y /= norm

    robot.set_velocity_vector(cmd_x, cmd_y)



    ########### 1. エンコーダとIMUを使った補完フィルタ ###########

    comp_loc.step()
    comp_x, comp_y, comp_theta = comp_loc.get_state()

    ############ 2. エンコーダによるホイールオドメトリ ###########

    odom.step()
    odom_x, odom_y, odom_theta = odom.get_state()

    ############ 3. IMUによる慣性航法 ###########

    imu_loc.update()
    imu_x, imu_y, imu_theta, _, _ = imu_loc.get_state()

    ########### 4. 真値の取得 ###########

    true_pos, true_orn = p.getBasePositionAndOrientation(robot_id)
    _, _, true_yaw = p.getEulerFromQuaternion(true_orn)

    ########### 5. 表示（小数点第2位） ###########

    print(f"X: ({true_pos[0]:.2f}, {comp_x:.2f}, {odom_x:.2f}, {imu_x:.2f})  Y: ({true_pos[1]:.2f}, {comp_y:.2f}, {odom_y:.2f}, {imu_y:.2f}) ")


    p.stepSimulation()
    time.sleep(dt)