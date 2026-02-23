import numpy as np
import math
import pybullet as p


class ComplementaryLocalization:

    def __init__(
        self,
        robot_id,
        left_joint,
        right_joint,
        wheel_radius,
        wheel_base,
        dt,
        alpha_yaw=0.01,
        alpha_vel=0.01,
        gyro_noise_std=0.02,
        accel_noise_std=0.1,
        encoder_noise_std=0.1,
        initial_x=0.0,
        initial_y=0.0,
        initial_theta=0.0,
    ):

        self.robot_id = robot_id
        self.left_joint = left_joint
        self.right_joint = right_joint

        self.r = wheel_radius
        self.L = wheel_base
        self.dt = dt

        self.alpha_yaw = alpha_yaw
        self.alpha_vel = alpha_vel

        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.encoder_noise_std = encoder_noise_std

        # --- 状態 ---
        self.theta = initial_theta
        self.vx = 0.0
        self.vy = 0.0
        self.x = initial_x
        self.y = initial_y

        # --- 高周波積分用 ---
        self.theta_gyro = initial_theta
        self.vx_acc = 0.0
        self.vy_acc = 0.0

        # --- 加速度差分用 ---
        self.prev_vx_world = 0.0
        self.prev_vy_world = 0.0

    # =========================================================
    # センサ
    # =========================================================

    def read_wheel_encoders(self):
        left_state = p.getJointState(self.robot_id, self.left_joint)
        right_state = p.getJointState(self.robot_id, self.right_joint)

        omega_L = -left_state[1] + np.random.normal(0, self.encoder_noise_std)
        omega_R = -right_state[1] + np.random.normal(0, self.encoder_noise_std)

        return omega_L, omega_R

    def read_gyro(self):
        _, base_ang_vel = p.getBaseVelocity(self.robot_id)
        wz = base_ang_vel[2] + np.random.normal(0, self.gyro_noise_std)
        return wz

    def read_accelerometer(self):

        lin_vel, _ = p.getBaseVelocity(self.robot_id)
        vx_world = lin_vel[0]
        vy_world = lin_vel[1]

        ax_world = (vx_world - self.prev_vx_world) / self.dt
        ay_world = (vy_world - self.prev_vy_world) / self.dt

        self.prev_vx_world = vx_world
        self.prev_vy_world = vy_world

        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        ax_body =  math.cos(yaw) * ax_world + math.sin(yaw) * ay_world
        ay_body = -math.sin(yaw) * ax_world + math.cos(yaw) * ay_world

        ax = ax_body + np.random.normal(0, self.accel_noise_std)
        ay = ay_body + np.random.normal(0, self.accel_noise_std)

        return ax, ay

    # =========================================================
    # Complementary Filter
    # =========================================================

    def step(self):

        omega_L, omega_R = self.read_wheel_encoders()
        wz = self.read_gyro()
        ax, ay = self.read_accelerometer()

        # --- オドメトリ ---
        v = self.r * 0.5 * (omega_L + omega_R)
        w_odom = self.r / self.L * (omega_R - omega_L)

        vx_odom = v * math.cos(self.theta)
        vy_odom = v * math.sin(self.theta)

        # --- IMU積分 ---
        self.theta_gyro += wz * self.dt
        self.vx_acc += ax * self.dt
        self.vy_acc += ay * self.dt

        # --- Yaw融合 ---
        theta_odom = self.theta + w_odom * self.dt

        self.theta = (
            self.alpha_yaw * self.theta_gyro +
            (1 - self.alpha_yaw) * theta_odom
        )

        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # --- 速度融合 ---
        self.vx = (
            self.alpha_vel * self.vx_acc +
            (1 - self.alpha_vel) * vx_odom
        )

        self.vy = (
            self.alpha_vel * self.vy_acc +
            (1 - self.alpha_vel) * vy_odom
        )

        # --- 位置更新 ---
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

    # =========================================================

    def get_state(self):
        return self.x, self.y, self.theta