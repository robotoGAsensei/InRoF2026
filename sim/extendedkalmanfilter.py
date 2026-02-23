# extendedkalmanfilter.py
import numpy as np
import math
import pybullet as p


class EKFLocalization:

    def __init__(
        self,
        robot_id,
        left_joint,
        right_joint,
        wheel_radius,
        wheel_base,
        dt,
        encoder_noise_std=0.1,
        yaw_noise_std=0.05,
        initial_x=0,
        initial_y=0,
        initial_theta=0,
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
        self.yaw_noise_std = abs(yaw_noise_std)

        # --- 状態 ---
        self.x = np.array([[initial_x], [initial_y], [initial_theta]])

        self.P = np.eye(3) * 0.01
        self.Q = np.diag([0.001, 0.001, 0.001])
        self.R = np.array([[0.01]])   # yawのみ観測

    # =========================================================
    # センサモデル
    # =========================================================

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

    def read_yaw_sensor(self):
        """
        真値取得 + ノイズ付与
        """
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw_true = p.getEulerFromQuaternion(orn)

        yaw_meas = yaw_true + np.random.normal(0, self.yaw_noise_std)

        return yaw_meas

    # =========================================================
    # EKF
    # =========================================================

    def predict(self, omega_L, omega_R):

        theta = self.x[2, 0]

        v = self.r * 0.5 * (omega_R + omega_L)
        w = self.r / self.L * (omega_R - omega_L)

        # --- 状態予測 ---
        self.x[0, 0] += v * math.cos(theta) * self.dt
        self.x[1, 0] += v * math.sin(theta) * self.dt
        self.x[2, 0] += w * self.dt

        # --- ヤコビアン ---
        F = np.array([
            [1, 0, -v * math.sin(theta) * self.dt],
            [0, 1,  v * math.cos(theta) * self.dt],
            [0, 0, 1]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, theta_meas):

        H = np.array([[0, 0, 1]])
        z = np.array([[theta_meas]])

        y = z - H @ self.x

        # 角度wrap（重要）
        y[0,0] = math.atan2(math.sin(y[0,0]), math.cos(y[0,0]))

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    # =========================================================
    # 外部呼び出し用ワンステップ
    # =========================================================

    def step(self):
        omega_L, omega_R = self.read_wheel_encoders()
        yaw_meas = self.read_yaw_sensor()

        self.predict(omega_L, omega_R)
        self.update(yaw_meas)

    def get_state(self):
        return self.x.flatten()