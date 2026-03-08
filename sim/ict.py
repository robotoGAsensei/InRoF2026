import math
import numpy as np

from lidar import Lidar          # 実機 LiDAR（pybullet rayTestBatch）
from lidar_odm import LidarOdm  # グリッドマップ LiDAR シミュレータ


class ICT:
    """LiDAR スキャンマッチング（距離差分二乗和最小化）。

    実機 Lidar のスキャン距離列と LidarOdm のスキャン距離列を比較し、
    差分の二乗和が最小となる (x, y, theta) を算出する。

    Parameters
    ----------
    lidar_odm : LidarOdm
        グリッドマップベースの LiDAR シミュレータ。
        map を保持しており scan(x, y, yaw) で距離列を生成する。
    """

    def __init__(self, lidar_odm: LidarOdm, min_hit_ratio: float = 0.5):
        self.lidar_odm    = lidar_odm
        self.min_hit_ratio = min_hit_ratio  # スキップ閾値（有効レイ比率の最低値）

    # ------------------------------------------------------------------
    # 内部コスト関数
    # ------------------------------------------------------------------

    def _cost(self, params, ref_distances, base_x, base_y, base_yaw):
        """(dx, dy, dtheta) に対する距離差分二乗和を返す。

        Parameters
        ----------
        params        : [dx, dy, dtheta]  探索変数
        ref_distances : array-like        実機 Lidar の距離列
        base_x, base_y, base_yaw : float  ベース姿勢（オドメトリ推定値）

        Returns
        -------
        float  sum_i (ref[i] - model[i])^2
        """
        dx, dy, dtheta = params
        model_dists, _, _ = self.lidar_odm.scan(
            base_x + dx,
            base_y + dy,
            base_yaw + dtheta,
        )
        diff = np.asarray(ref_distances) - np.asarray(model_dists)
        return float(np.dot(diff, diff))

    # ------------------------------------------------------------------
    # 数値偏微分（中心差分）
    # ------------------------------------------------------------------

    def _gradient(self, dx, dy, dtheta, ref_distances, base_x, base_y, base_yaw,
                  h_xy=1e-3, h_th=1e-2):
        """コスト関数の (dx, dy, dtheta) に関する偏微分を中心差分で求める。

        Returns
        -------
        (gx, gy, gtheta) : float  各変数に関する偏微分値
        """
        gx = (self._cost([dx + h_xy, dy,         dtheta        ], ref_distances, base_x, base_y, base_yaw)
            - self._cost([dx - h_xy, dy,         dtheta        ], ref_distances, base_x, base_y, base_yaw)) / (2 * h_xy)
        gy = (self._cost([dx,        dy + h_xy,  dtheta        ], ref_distances, base_x, base_y, base_yaw)
            - self._cost([dx,        dy - h_xy,  dtheta        ], ref_distances, base_x, base_y, base_yaw)) / (2 * h_xy)
        gt = (self._cost([dx,        dy,         dtheta + h_th ], ref_distances, base_x, base_y, base_yaw)
            - self._cost([dx,        dy,         dtheta - h_th ], ref_distances, base_x, base_y, base_yaw)) / (2 * h_th)
        return gx, gy, gt

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def match(self, ref_distances, base_x, base_y, base_yaw,
              lr=1.0e-4, maxiter=10):
        """スキャンマッチングを実行し最適姿勢 (x, y, theta) を返す。

        誤差の二乗和を x, y, theta で偏微分し、誤差が減る方向へ
        学習率 lr だけ修正することを maxiter 回繰り返す。

        Parameters
        ----------
        ref_distances : list[float]
            実機 Lidar.scan() から得た距離列。
        base_x   : float  オドメトリ推定 X 座標 [m]
        base_y   : float  オドメトリ推定 Y 座標 [m]
        base_yaw : float  オドメトリ推定 yaw 角 [rad]
        lr       : float  学習率（既定 1e-4）
        maxiter  : int    イテレーション回数（既定 200）

        Returns
        -------
        best_x   : float       最適 X 座標 [m]
        best_y   : float       最適 Y 座標 [m]
        best_yaw : float       最適 yaw 角 [rad]
        result   : dict | None
            {'iters': int}。スキップ時は None。
        """
        # 有効レイ（max_dist 未満でヒットしたレイ）の比率チェック
        max_dist  = self.lidar_odm.MAX_DIST
        hit_count = sum(1 for d in ref_distances if d < max_dist)
        hit_ratio = hit_count / len(ref_distances)
        if hit_ratio < self.min_hit_ratio:
            return base_x, base_y, base_yaw, {"hit_ratio": hit_ratio, "skipped": True}

        cost_init = self._cost([0.0, 0.0, 0.0], ref_distances, base_x, base_y, base_yaw)

        dx, dy, dtheta = 0.0, 0.0, 0.0
        gx, gy, gt = 0.0, 0.0, 0.0
        for _ in range(maxiter):
            gx, gy, gt = self._gradient(dx, dy, dtheta, ref_distances,
                                        base_x, base_y, base_yaw)
            dx     -= lr * gx
            dy     -= lr * gy
            dtheta -= lr * gt

        cost_final = self._cost([dx, dy, dtheta], ref_distances, base_x, base_y, base_yaw)

        best_x   = base_x   + dx
        best_y   = base_y   + dy
        best_yaw = base_yaw + dtheta

        return best_x, best_y, best_yaw, {
            "hit_ratio":  hit_ratio,
            "skipped":    False,
            "cost_init":  cost_init,
            "cost_final": cost_final,
            "grad":       (gx, gy, gt),
            "correction": (dx, dy, dtheta),
        }

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def distances_to_points(distances, ray_from, ray_to, max_dist=None):
        """距離列とレイ情報からヒット点の 2D 座標列を返す。

        max_dist を指定すると、それ未満のヒット点のみを返す
        （マックス距離に達した「ミス」レイを除外）。

        Parameters
        ----------
        distances : list[float]
        ray_from  : list[[x, y, z]]
        ray_to    : list[[x, y, z]]
        max_dist  : float, optional

        Returns
        -------
        points : np.ndarray  shape (N, 2)  ヒット点の (x, y) 配列
        """
        points = []
        for d, f, t in zip(distances, ray_from, ray_to):
            if max_dist is not None and d >= max_dist:
                continue
            dx = t[0] - f[0]
            dy = t[1] - f[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            px = f[0] + dx / length * d
            py = f[1] + dy / length * d
            points.append([px, py])
        return np.array(points) if points else np.empty((0, 2))
