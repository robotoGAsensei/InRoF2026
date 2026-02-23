# odometry.py
import math
import numpy as np
import pybullet as p


class Odometry:
    def __init__(
        self,
        robot_id,
        left_joint,
        right_joint,
        wheel_radius,
        wheel_base,
        dt,
        encoder_noise_std=0.1,
        initial_x=0.0,
        initial_y=0.0,
        initial_theta=0.0
    ):

        # --- PyBullet関連 ---
        self.robot_id = robot_id
        self.left_joint = left_joint
        self.right_joint = right_joint

        # --- 物理パラメータ ---
        self.r = wheel_radius
        self.L = wheel_base
        self.dt = dt

        # --- ノイズ ---
        self.encoder_noise_std = abs(encoder_noise_std)

        # --- 状態 ---
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta

    # =====================================================
    # センサモデル（エンコーダ）
    # =====================================================

    def read_wheel_encoders(self):
        """
        真値取得 + ノイズ付与
        """

        left_state = p.getJointState(self.robot_id, self.left_joint)
        right_state = p.getJointState(self.robot_id, self.right_joint)

        omega_L_true = -left_state[1]
        omega_R_true = -right_state[1]

        omega_L = omega_L_true + np.random.normal(0, self.encoder_noise_std)
        omega_R = omega_R_true + np.random.normal(0, self.encoder_noise_std)

        return omega_L, omega_R

    # =====================================================
    # オドメトリ更新
    # =====================================================

    def update(self, omega_L, omega_R):

        v = self.r * 0.5 * (omega_R + omega_L)
        w = self.r / self.L * (omega_R - omega_L)

        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta += w * self.dt

    # =====================================================
    # 外部呼び出し用
    # =====================================================

    def step(self):
        omega_L, omega_R = self.read_wheel_encoders()
        self.update(omega_L, omega_R)

    def get_state(self):
        return self.x, self.y, self.theta