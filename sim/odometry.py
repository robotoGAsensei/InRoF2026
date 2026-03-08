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
        slip_base_L=0.02,       # 左輪の定常スリップ率 (0〜1)
        slip_base_R=0.02,       # 右輪の定常スリップ率 (0〜1)
        slip_vel_coeff=0.001,   # 速度比例スリップ係数（|omega| に乗算）
        slip_event_prob=0.001,  # 突発スリップイベントの発生確率 (per step)
        slip_event_mag=0.25,    # 突発スリップイベント時の追加スリップ量の上限
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

        # --- スリップモデルパラメータ ---
        self.slip_base_L     = slip_base_L
        self.slip_base_R     = slip_base_R
        self.slip_vel_coeff  = slip_vel_coeff
        self.slip_event_prob = slip_event_prob
        self.slip_event_mag  = slip_event_mag

        # --- 状態 ---
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta

    # =====================================================
    # センサモデル（エンコーダ＋車輪スリップ）
    # =====================================================

    def read_wheel_encoders(self):
        """
        真値取得 + 車輪スリップモデル

        スリップ発生時、車輪は地面接触点の移動量よりも多く回転する。
        エンコーダは車輪回転を計測するため omega_measured > omega_ground となり、
        オドメトリは実際の移動量を過大推定してドリフトが発生する。

        スリップ率の内訳:
          定常スリップ   : 路面・タイヤ特性による非対称な定常誤差
          速度依存スリップ: 高速回転ほどスリップが増加
          突発スリップ   : 路面の局所的な段差・砂などによる確率的なスリップ
        """
        left_state  = p.getJointState(self.robot_id, self.left_joint)
        right_state = p.getJointState(self.robot_id, self.right_joint)

        omega_L_true = -left_state[1]
        omega_R_true = -right_state[1]

        # 速度依存スリップ率（回転速度が大きいほどスリップ増加）
        slip_L = self.slip_base_L + self.slip_vel_coeff * abs(omega_L_true)
        slip_R = self.slip_base_R + self.slip_vel_coeff * abs(omega_R_true)

        # 突発スリップイベント（路面の乱れなどをモデル化）
        if np.random.random() < self.slip_event_prob:
            slip_L += self.slip_event_mag * np.random.random()
        if np.random.random() < self.slip_event_prob:
            slip_R += self.slip_event_mag * np.random.random()

        # スリップ分だけエンコーダが過剰な回転を計測する
        omega_L = omega_L_true * (1.0 + slip_L)
        omega_R = omega_R_true * (1.0 + slip_R)

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

    def set_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta