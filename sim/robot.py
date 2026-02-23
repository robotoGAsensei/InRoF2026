import pybullet as p

class DifferentialRobot:

    def __init__(self, body_id, left_joint, right_joint,
                 wheel_base=0.1, wheel_radius=0.03):
        self.body = body_id
        self.left = left_joint
        self.right = right_joint
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius

    def set_velocity_vector(self, vx, vy):

        # ベクトル → 差動駆動変換
        v = vx * 0.3
        omega = vy * 5.0

        left_speed  = (v - omega * self.wheel_base/2) / self.wheel_radius
        right_speed = (v + omega * self.wheel_base/2) / self.wheel_radius

        p.setJointMotorControl2(self.body, self.left,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_speed,
                                force=5)

        p.setJointMotorControl2(self.body, self.right,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_speed,
                                force=5)