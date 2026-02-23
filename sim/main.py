import pybullet as p
import pybullet_data
import time
import math

from util import get_joint_index, get_link_index
from robot import DifferentialRobot
from lidar import Lidar
from obstacle_avoidance import ObstacleAvoidance

# ===== 初期化 =====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("field.urdf", [0, 0, 0], useFixedBase=True)

robot_id = p.loadURDF("robot.urdf", [0.25, -0.25, 0.55])

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

    p.stepSimulation()
    time.sleep(1./240.)