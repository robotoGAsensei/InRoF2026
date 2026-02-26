# mcl.py

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
        self.motion_noise_std = 0.02
        self.sensor_noise_std = 0.05

        # Lidar parameters（既存 lidar と揃える）
        self.FOV = math.radians(85)
        self.NUM_RAYS = 60
        self.MAX_DIST = 1.0


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

            self.particles[i] = [x, y, theta]


    # ==================================================
    # Sensor Update
    # ==================================================
    def sensor_update(self, real_distances):

        log_weights = np.zeros(self.N)

        for i in range(self.N):

            x, y, theta = self.particles[i]
            log_weight = 0.0

            for j in range(self.NUM_RAYS):

                angle = -self.FOV/2 + self.FOV * j/(self.NUM_RAYS-1)
                world_angle = theta + angle

                ray_from = [x, y, self.lidar_height]

                ray_to = [
                    x + math.cos(world_angle) * self.MAX_DIST,
                    y + math.sin(world_angle) * self.MAX_DIST,
                    self.lidar_height
                ]

                # rayTestは、ray_fromからray_toに向かってレイを飛ばし、最初に衝突するオブジェクトとの距離を返す関数
                # 既知マップ(field.urdf)と仮説位置から計算される距離を得るために使用するpybulletの機能
                # 実際のロボットではrayTest相当の機能を実装する必要がある
                result = p.rayTest(ray_from, ray_to)[0]

                # 仮説位置をたくさん用意し
                # それぞれについて「その位置ならこう見えるはず」を計算したのがsim_dist
                # sim_distは既知マップと仮説位置から計算される距離
                if result[2] < 1.0:
                    sim_dist = result[2] * self.MAX_DIST
                else:
                    sim_dist = self.MAX_DIST

                # --- MAXレンジは情報量が低いので除外 ---
                if real_distances[j] >= self.MAX_DIST * 0.99:
                    continue
                if sim_dist >= self.MAX_DIST * 0.99:
                    continue

                error = real_distances[j] - sim_dist

                # 仮説位置の確からしさを計算
                # 実際の位置が10cmだとして、仮説位置が5cmの場合に、
                # sim_dist=5であっても観測値としては10付近が観測されるのでerrorが大きくなる
                # つまり仮説位置が5cmの場合に10cm付近が観測される確率は小さい事を表している
                # RAYの数が60本なら60本分の確率を掛け合わせて ※
                # 作った重み係数は、その仮説位置の総合的な確からしさを表す指標になる
                #
                # もしくは、
                # 仮説位置 x_i が正しいと仮定したときに、
                # 現在のLidar観測全体が得られる確率を計算している。
                # rayごとの尤度を掛け合わせることで ※
                # 粒子の総合的な尤度（重み）を求めている。
                # ※途中計算はlog空間で行うため実装上は足し算になっている
                log_weight += -(error**2) / (2 * self.sensor_noise_std**2)

            log_weights[i] = log_weight

        # ==============================
        # log-sum-exp 安定化
        # ==============================

        max_log = np.max(log_weights)
        log_weights -= max_log

        # 途中計算はlog空間で行い、最後に指数を取ることで数値の安定性を保つ
        weights = np.exp(log_weights)

        # ゼロ割り防止
        weights += 1e-300
        weights /= np.sum(weights)

        self.weights = weights

    # ==================================================
    # Resampling　「確からしい粒子を増やし、あり得ない粒子を消す」操作
    # ==================================================
    def resample(self):

        # 確率分布の累積和
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