import math

class ObstacleAvoidance:

    def __init__(self, fov, num_rays, safe_dist=0.15):
        self.FOV = fov
        self.NUM_RAYS = num_rays
        self.SAFE_DIST = safe_dist

    def compute_avoid_vector(self, distances):

        avoid_x = 0.0
        avoid_y = 0.0

        for i, d in enumerate(distances):

            if d < self.SAFE_DIST:

                angle = -self.FOV/2 + self.FOV * i/(self.NUM_RAYS-1)

                obs_x = math.cos(angle)
                obs_y = math.sin(angle)

                # 距離に比例した重み
                weight = (self.SAFE_DIST - d) / self.SAFE_DIST

                # 反発ベクトル
                avoid_x -= obs_x * weight
                avoid_y -= obs_y * weight

        return -avoid_x, -avoid_y