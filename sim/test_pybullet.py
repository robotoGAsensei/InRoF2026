import pybullet as p
import pybullet_data
import time
import math

# ================= 初期化 =================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("field.urdf", [0, 0, 0], useFixedBase=True)

robot = p.loadURDF("robot.urdf", [0.25, -0.25, 0.55])

# ================= ユーティリティ =================
def get_joint_index(body, joint_name):
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        if info[1].decode("utf-8") == joint_name:
            return i
    return None

def get_link_index(body, link_name):
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return None

left_wheel = get_joint_index(robot, "left_wheel_joint")
right_wheel = get_joint_index(robot, "right_wheel_joint")
lidar_link_index = get_link_index(robot, "lidar_link")

if lidar_link_index is None:
    raise RuntimeError("lidar_link が見つかりません")

# ================= LiDAR設定 =================
FOV = math.radians(100)
NUM_RAYS = 60
MAX_DIST = 0.3
SAFE_DIST = 0.15

debug_line_ids = [-1] * NUM_RAYS

# ================= LiDAR取得 =================
def get_lidar_data():

    link_state = p.getLinkState(robot, lidar_link_index)
    base_pos = link_state[0]
    base_orn = link_state[1]

    rot = p.getMatrixFromQuaternion(base_orn)

    ray_from = []
    ray_to = []

    for i in range(NUM_RAYS):
        angle = -FOV/2 + FOV * i/(NUM_RAYS-1)

        dx = math.cos(angle)
        dy = math.sin(angle)

        world_dx = rot[0]*dx + rot[1]*dy
        world_dy = rot[3]*dx + rot[4]*dy

        from_pos = [base_pos[0], base_pos[1], base_pos[2]]
        to_pos = [
            base_pos[0] + world_dx * MAX_DIST,
            base_pos[1] + world_dy * MAX_DIST,
            base_pos[2]
        ]

        ray_from.append(from_pos)
        ray_to.append(to_pos)

    results = p.rayTestBatch(ray_from, ray_to)

    distances = []
    for r in results:
        hit_fraction = r[2]
        if hit_fraction < 1.0:
            distances.append(hit_fraction * MAX_DIST)
        else:
            distances.append(MAX_DIST)

    return distances, ray_from, ray_to, results


# ================= メインループ =================
wheel_base = 0.1
wheel_radius = 0.03

while True:

    # ===== LiDAR取得 =====
    distances, ray_from, ray_to, results = get_lidar_data()

    # ===== 可視化（軽量） =====
    for i, r in enumerate(results):
        color = [1, 0, 0] if r[2] < 1 else [0, 1, 0]
        debug_line_ids[i] = p.addUserDebugLine(
            ray_from[i],
            ray_to[i],
            color,
            1,
            replaceItemUniqueId=debug_line_ids[i]
        )

    # ===== ユーザー入力 =====
    keys = p.getKeyboardEvents()

    vx = 0
    vy = 0
    user_speed = 0.2

    if ord('i') in keys:
        vx = user_speed
        print("前進")
    if ord('k') in keys:
        vx = -user_speed
        print("後退")
    if ord('j') in keys:
        vy = user_speed
        print("左移動")
    if ord('l') in keys:
        vy = -user_speed
        print("右移動")

    # ===== 回避ベクトル生成 =====
    avoid_x = 0
    avoid_y = 0

    for i, d in enumerate(distances):
        if d < SAFE_DIST:
            angle = -FOV/2 + FOV * i/(NUM_RAYS-1)

            obs_x = math.cos(angle)
            obs_y = math.sin(angle)

            weight = (SAFE_DIST - d) / SAFE_DIST

            avoid_x -= obs_x * weight
            avoid_y -= obs_y * weight

    # ===== ベクトル合成 =====
    cmd_x = vx - avoid_x
    cmd_y = vy - avoid_y

    # 正規化（暴走防止）
    norm = math.hypot(cmd_x, cmd_y)
    if norm > 1e-5:
        cmd_x /= norm
        cmd_y /= norm

    # ===== 差動駆動変換 =====
    v = cmd_x * 0.3
    omega = cmd_y * 5.0

    left_speed  = (v - omega * wheel_base/2) / wheel_radius
    right_speed = (v + omega * wheel_base/2) / wheel_radius

    p.setJointMotorControl2(robot, left_wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=left_speed,
                            force=5)

    p.setJointMotorControl2(robot, right_wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=right_speed,
                            force=5)

    p.stepSimulation()
    time.sleep(1./240.)