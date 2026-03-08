import math
import numpy as np
from scipy.optimize import minimize

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

    def __init__(self, lidar_odm: LidarOdm):
        self.lidar_odm = lidar_odm

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
    # 公開 API
    # ------------------------------------------------------------------

    def match(self, ref_distances, base_x, base_y, base_yaw,
              x0=None, method="Nelder-Mead", options=None):
        """スキャンマッチングを実行し最適姿勢 (x, y, theta) を返す。

        Parameters
        ----------
        ref_distances : list[float]
            実機 Lidar.scan() から得た距離列。
        base_x   : float  オドメトリ推定 X 座標 [m]
        base_y   : float  オドメトリ推定 Y 座標 [m]
        base_yaw : float  オドメトリ推定 yaw 角 [rad]
        x0       : array-like, optional
            最適化初期値 [dx, dy, dtheta]。省略時は [0, 0, 0]。
        method   : str
            scipy.optimize.minimize の最適化手法。
            勾配不要の "Nelder-Mead" を既定とする。
        options  : dict, optional
            scipy.optimize.minimize に渡すオプション。

        Returns
        -------
        best_x   : float          最適 X 座標 [m]
        best_y   : float          最適 Y 座標 [m]
        best_yaw : float          最適 yaw 角 [rad]
        result   : OptimizeResult scipy の最適化結果オブジェクト
        """
        if x0 is None:
            x0 = [0.0, 0.0, 0.0]
        if options is None:
            options = {"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000}

        result = minimize(
            self._cost,
            x0,
            args=(ref_distances, base_x, base_y, base_yaw),
            method=method,
            options=options,
        )

        dx, dy, dtheta = result.x
        best_x   = base_x   + dx
        best_y   = base_y   + dy
        best_yaw = base_yaw + dtheta

        return best_x, best_y, best_yaw, result

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
