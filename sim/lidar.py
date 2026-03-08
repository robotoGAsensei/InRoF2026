import pybullet as p
import math


class Lidar:

    def __init__(self, robot_id, link_index,
                 fov=math.radians(85),
                 num_rays=60,
                 max_dist=1.0,
                 lidar_height=0.03):   # ← 高さ固定追加

        self.robot = robot_id
        self.link = link_index
        self.FOV = fov
        self.NUM_RAYS = num_rays
        self.MAX_DIST = max_dist
        self.LIDAR_HEIGHT = lidar_height
        self.debug_ids = [-1] * num_rays

    def scan(self):

        # LiDARの位置取得（x,yのみ使用）
        link_state = p.getLinkState(self.robot, self.link)
        base_pos = link_state[0]

        # ロボットの姿勢取得
        _, base_orn = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw = p.getEulerFromQuaternion(base_orn)

        # yawのみ使用（roll/pitch無視）
        flat_orn = p.getQuaternionFromEuler([0, 0, yaw])
        rot = p.getMatrixFromQuaternion(flat_orn)

        ray_from = []
        ray_to = []

        # 画角内に均等にレイを配置してscan
        for i in range(self.NUM_RAYS):

            angle = -self.FOV / 2 + self.FOV * i / (self.NUM_RAYS - 1)

            dx = math.cos(angle)
            dy = math.sin(angle)

            # yaw回転適用
            world_dx = rot[0] * dx + rot[1] * dy
            world_dy = rot[3] * dx + rot[4] * dy

            # 高さ固定（完全2D化）
            from_pos = [
                base_pos[0],
                base_pos[1],
                self.LIDAR_HEIGHT
            ]

            to_pos = [
                base_pos[0] + world_dx * self.MAX_DIST,
                base_pos[1] + world_dy * self.MAX_DIST,
                self.LIDAR_HEIGHT
            ]

            ray_from.append(from_pos)
            ray_to.append(to_pos)

        # rayTestBatchの結果はリストで、各要素は以下のタプル構造を持つ
        # results[i] = (
        #     objectUniqueId,   # [0] ヒットしたオブジェクトのID（ミス時は -1）
        #     linkIndex,        # [1] ヒットしたリンクのインデックス（ミス時は -1）
        #     hitFraction,      # [2] レイ長さに対するヒット位置の割合 0.0〜1.0（ミス時は 1.0）
        #     hitPosition,      # [3] ヒット座標 (x, y, z)（ミス時は (0, 0, 0)）
        #     hitNormal,        # [4] ヒット面の法線ベクトル (x, y, z)（ミス時は (0, 0, 0)）
        # )
        results = p.rayTestBatch(ray_from, ray_to)

        distances = []

        for r in results:
            hit_fraction = r[2]
            if hit_fraction < 1.0:
                distances.append(hit_fraction * self.MAX_DIST)
            else:
                distances.append(self.MAX_DIST)

        return distances, ray_from, ray_to, results

    def draw(self, ray_from, ray_to, results):

        for i, r in enumerate(results):
            color = [1, 0, 0] if r[2] < 1 else [0, 1, 0]

            self.debug_ids[i] = p.addUserDebugLine(
                ray_from[i],
                ray_to[i],
                color,
                1,
                replaceItemUniqueId=self.debug_ids[i]
            )