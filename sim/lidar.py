import pybullet as p
import math

class Lidar:

    def __init__(self, robot_id, link_index,
                 fov=math.radians(85),
                 num_rays=60,
                 max_dist=1.0):

        self.robot = robot_id
        self.link = link_index
        self.FOV = fov
        self.NUM_RAYS = num_rays
        self.MAX_DIST = max_dist
        self.debug_ids = [-1] * num_rays

    def scan(self):

        link_state = p.getLinkState(self.robot, self.link)
        base_pos = link_state[0]
        base_orn = link_state[1]

        rot = p.getMatrixFromQuaternion(base_orn)

        ray_from = []
        ray_to = []

        for i in range(self.NUM_RAYS):
            angle = -self.FOV/2 + self.FOV * i/(self.NUM_RAYS-1)

            dx = math.cos(angle)
            dy = math.sin(angle)

            world_dx = rot[0]*dx + rot[1]*dy
            world_dy = rot[3]*dx + rot[4]*dy

            from_pos = base_pos
            to_pos = [
                base_pos[0] + world_dx * self.MAX_DIST,
                base_pos[1] + world_dy * self.MAX_DIST,
                base_pos[2]
            ]

            ray_from.append(from_pos)
            ray_to.append(to_pos)

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