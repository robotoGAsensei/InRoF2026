import numpy as np
import math
import pybullet as p


class MCL:

    def __init__(self,
                 num_particles,
                 robot_id,
                 lidar_link,
                 wheel_radius,
                 wheel_base,
                 dt,
                 init_x,
                 init_y,
                 init_theta):

        self.N = num_particles
        self.robot_id = robot_id
        self.lidar_link = lidar_link
        self.r = wheel_radius
        self.L = wheel_base
        self.dt = dt

        link_state = p.getLinkState(self.robot_id, self.lidar_link)
        self.lidar_height = link_state[0][2]

        # パーティクル初期化（初期位置は既知なので小さな分散）
        self.particles = np.zeros((self.N, 3))
        self.particles[:, 0] = np.random.normal(init_x, 0.02, self.N)
        self.particles[:, 1] = np.random.normal(init_y, 0.02, self.N)
        self.particles[:, 2] = np.random.normal(init_theta, 0.05, self.N)

        self.weights = np.ones(self.N) / self.N

        # センサノイズ
        self.motion_noise_std = 0.05
        self.sensor_noise_std = 0.2   # ← 少し緩めた（重要）

        # Lidar parameters（既存 lidar と揃える）
        self.FOV = math.radians(180)
        self.MAX_DIST = 2.0

    # ==================================================
    # Motion Update
    # ==================================================
    def motion_update(self, omega_L, omega_R):

        v = self.r * 0.5 * (omega_R + omega_L)
        w = self.r / self.L * (omega_R - omega_L)

        for i in range(self.N):

            noisy_v = v + np.random.normal(0, self.motion_noise_std)
            noisy_w = w + np.random.normal(0, self.motion_noise_std)

            x, y, theta = self.particles[i]

            x += noisy_v * math.cos(theta) * self.dt
            y += noisy_v * math.sin(theta) * self.dt
            theta += noisy_w * self.dt

            # 角度正規化（重要）
            theta = math.atan2(math.sin(theta), math.cos(theta))

            self.particles[i] = [x, y, theta]


    # ==================================================
    # Sensor Update
    # ==================================================
    def sensor_update(self, real_distances):

        log_weights = np.zeros(self.N)
        num_rays = len(real_distances)

        for i in range(self.N):

            x, y, theta = self.particles[i]

            ray_from_list = []
            ray_to_list = []
            valid_indices = []

            # -------------------------------
            # 有効レイだけ抽出
            # -------------------------------
            for j in range(num_rays):

                if real_distances[j] >= self.MAX_DIST * 0.99:
                    continue

                angle = -self.FOV/2 + self.FOV * j/(num_rays-1)
                world_angle = theta + angle

                ray_from = [x, y, self.lidar_height]
                ray_to = [
                    x + math.cos(world_angle) * self.MAX_DIST,
                    y + math.sin(world_angle) * self.MAX_DIST,
                    self.lidar_height
                ]

                ray_from_list.append(ray_from)
                ray_to_list.append(ray_to)
                valid_indices.append(j)

            # 有効レイが無い場合
            if len(valid_indices) == 0:
                log_weights[i] = 0.0
                continue

            # -------------------------------
            # rayTestBatchで高速化
            # -------------------------------
            results = p.rayTestBatch(ray_from_list, ray_to_list)

            error_sum = 0.0

            for k, result in enumerate(results):

                if result[2] < 1.0:
                    sim_dist = result[2] * self.MAX_DIST
                else:
                    sim_dist = self.MAX_DIST

                if sim_dist >= self.MAX_DIST * 0.99:
                    continue

                real_d = real_distances[valid_indices[k]]
                error = real_d - sim_dist

                error_sum += error**2

            # -------------------------------
            # ★ ここが最重要修正 ★
            # レイ数で平均化して鋭さを抑える
            # -------------------------------
            error_mean = error_sum / len(valid_indices)

            log_weights[i] = -(error_mean) / (2 * self.sensor_noise_std**2)

        # ==============================
        # log-sum-exp 安定化
        # ==============================

        max_log = np.max(log_weights)
        log_weights -= max_log

        weights = np.exp(log_weights)

        # ゼロ割り防止
        weights += 1e-300
        weights /= np.sum(weights)

        self.weights = weights


    # ==================================================
    # Resampling　「確からしい粒子を増やし、あり得ない粒子を消す」操作
    # ==================================================
    def resample(self):

        ess = 1.0 / np.sum(self.weights ** 2)

        # 粒子の半分以下になったらリサンプリング
        if ess > self.N / 2:
            return

        cumulative = np.cumsum(self.weights)
        step = 1.0 / self.N
        r = np.random.uniform(0, step)

        new_particles = []
        i = 0

        for m in range(self.N):
            U = r + m * step
            while U > cumulative[i]:
                i += 1
            new_particles.append(self.particles[i])

        self.particles = np.array(new_particles)
        self.weights.fill(1.0 / self.N)

    # ==================================================
    # 推定値（角度を正しく平均）
    # ==================================================
    def estimate(self):

        # x, y は通常の重み付き平均
        x_mean = np.average(self.particles[:, 0], weights=self.weights)
        y_mean = np.average(self.particles[:, 1], weights=self.weights)

        # θ は円周平均
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta_mean = np.arctan2(sin_sum, cos_sum)

        return np.array([x_mean, y_mean, theta_mean])