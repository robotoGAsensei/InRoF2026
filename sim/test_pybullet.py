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

# ================= ジョイント取得 =================
left_wheel = get_joint_index(robot, "left_wheel_joint")
right_wheel = get_joint_index(robot, "right_wheel_joint")
lidar_link_index = get_link_index(robot, "lidar_link")

print("left wheel:", left_wheel)
print("right wheel:", right_wheel)
print("lidar link:", lidar_link_index)

if lidar_link_index is None:
    raise RuntimeError("lidar_link が見つかりません。URDFを確認してください。")

# ================= LiDAR設定 =================
FOV = math.radians(100)
NUM_RAYS = 60
MAX_DIST = 0.3
debug_line_ids = [-1] * NUM_RAYS

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

        # ローカル→ワールド変換
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


# ================= 障害物回避 =================
def obstacle_avoidance(distances):

    left = distances[:NUM_RAYS//3]
    center = distances[NUM_RAYS//3:2*NUM_RAYS//3]
    right = distances[2*NUM_RAYS//3:]

    min_center = min(center)
    speed = 8

    if min_center < 0.15:
        if min(left) > min(right):
            return -speed, speed
        else:
            return speed, -speed
    else:
        return speed, speed


# ================= メインループ =================
while True:

    distances, ray_from, ray_to, results = get_lidar_data()

    for i, r in enumerate(results):
        color = [1, 0, 0] if r[2] < 1 else [0, 1, 0]

        debug_line_ids[i] = p.addUserDebugLine(
            ray_from[i],
            ray_to[i],
            color,
            1,
            replaceItemUniqueId=debug_line_ids[i]
        )

    left_speed, right_speed = obstacle_avoidance(distances)

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