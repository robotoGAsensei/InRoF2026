import pybullet as p
import pybullet_data
import time
import math
import mcl
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

mcl = mcl.MCL(
    num_particles=30,
    robot_id=robot_id,
    lidar_link=lidar_link,
    wheel_radius=wheel_radius,
    wheel_base=wheel_base,
    dt=dt,
    init_x=init_pos[0],
    init_y=init_pos[1],
    init_theta=init_yaw
)

def angle_diff(a, b):
    """
    角度差を -π〜π に正規化
    """
    return np.arctan2(np.sin(a - b), np.cos(a - b))

# ===== メインループ =====
while True:

    distances, ray_from, ray_to, results = lidar.scan()
    # lidar.draw(ray_from, ray_to, results)

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

    # ===== オドメトリ更新 =====
    odom.step()
    odom_x, odom_y, odom_theta = odom.get_state()

    # ===== MCL更新 =====
    omega_L, omega_R = odom.read_wheel_encoders()
    mcl.motion_update(omega_L, omega_R)
    mcl.sensor_update(distances)
    mcl.resample()
    mcl_x, mcl_y, mcl_theta = mcl.estimate()

    # ===== 真値取得 =====
    true_pos, true_orn = p.getBasePositionAndOrientation(robot_id)
    _, _, true_theta = p.getEulerFromQuaternion(true_orn)

    # ===== 表示 =====
    theta_err_odom = angle_diff(true_theta, odom_theta)
    theta_err_mcl  = angle_diff(true_theta, mcl_theta)

    print(
        f"X: ({true_pos[0]:.2f}, {odom_x:.2f}, {mcl_x:.2f})  "
        f"Y: ({true_pos[1]:.2f}, {odom_y:.2f}, {mcl_y:.2f})  "
        f"θ: ({true_theta:.2f}, {odom_theta:.2f}, {mcl_theta:.2f})  "
        f"θerr(O:{theta_err_odom:.3f}, M:{theta_err_mcl:.3f})"
    )
    p.stepSimulation()
    time.sleep(dt)