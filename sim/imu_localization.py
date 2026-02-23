# imu_localization.py
import math
import numpy as np
import pybullet as p


class IMULocalization:
    def __init__(
        self,
        robot_id,
        dt,
        accel_noise_std=0.05,
        gyro_noise_std=0.01,
        initial_x=0.0,
        initial_y=0.0,
        initial_theta=0.0,
    ):
        self.robot_id = robot_id
        self.dt = dt

        # 状態
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta
        self.vx = 0.0
        self.vy = 0.0

        # ノイズ
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std

        # 前回速度（加速度計算用）
        self.prev_lin_vel = np.zeros(3)

    # ==============================
    # センサモデル
    # ==============================
    def read_imu(self):
        """
        PyBulletから真値取得 → IMU値を生成
        戻り値:
            ax_body, ay_body, yaw
        """

        # --- 真値取得 ---
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)

        lin_vel = np.array(lin_vel)

        # --- 加速度計算（world frame） ---
        acc_world = (lin_vel - self.prev_lin_vel) / self.dt
        self.prev_lin_vel = lin_vel.copy()

        # --- 姿勢取得 ---
        roll, pitch, yaw_true = p.getEulerFromQuaternion(orn)

        # --- world → body 変換（2D想定） ---
        c = math.cos(yaw_true)
        s = math.sin(yaw_true)

        ax_body =  c * acc_world[0] + s * acc_world[1]
        ay_body = -s * acc_world[0] + c * acc_world[1]

        # --- ノイズ付与 ---
        ax_body += np.random.normal(0, self.accel_noise_std)
        ay_body += np.random.normal(0, self.accel_noise_std)
        yaw_meas = yaw_true + np.random.normal(0, self.gyro_noise_std)

        return ax_body, ay_body, yaw_meas

    # ==============================
    # 慣性航法更新
    # ==============================
    def update(self):
        ax_body, ay_body, yaw = self.read_imu()

        # body → world 変換
        c = math.cos(yaw)
        s = math.sin(yaw)

        ax_world = c * ax_body - s * ay_body
        ay_world = s * ax_body + c * ay_body

        # 速度更新
        self.vx += ax_world * self.dt
        self.vy += ay_world * self.dt

        # 位置更新
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        self.theta = yaw

    # ==============================
    # 状態取得
    # ==============================
    def get_state(self):
        return self.x, self.y, self.theta, self.vx, self.vy