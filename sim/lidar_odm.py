import math
import numpy as np


class LidarOdm:
    """Occupancy grid ベースの LiDAR シミュレータ（オドメトリ姿勢使用）。

    rayTestBatch を使わず、occupancy grid 上でレイマーチングを行い
    distances / ray_from / ray_to を返す。

    Parameters
    ----------
    grid       : array-like (H x W)  occupancy grid（1=障害物, 0=空き）
    resolution : float               グリッド解像度 [m/cell]
    origin     : (float, float)      grid[0][0] のワールド座標 (ox, oy) [m]
    fov        : float               視野角 [rad]  (default 85°)
    num_rays   : int                 レイ本数       (default 60)
    max_dist   : float               最大検出距離 [m] (default 1.0)
    lidar_offset_x : float           ロボット前方へのオフセット [m]
                                     robot.urdf lidar_joint xyz="0.055 0 0"
    lidar_height   : float           Z高さ [m]（可視化用）
    """

    def __init__(self, grid, resolution, origin,
                 fov=math.radians(85),
                 num_rays=60,
                 max_dist=1.0,
                 lidar_offset_x=0.055,
                 lidar_height=0.03):

        self.grid = np.asarray(grid)
        self.resolution = resolution
        self.origin = origin
        self.FOV = fov
        self.NUM_RAYS = num_rays
        self.MAX_DIST = max_dist
        self.lidar_offset_x = lidar_offset_x
        self.lidar_height = lidar_height

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _world_to_cell(self, wx, wy):
        """ワールド座標 → グリッドセル (col, row)"""
        col = int((wx - self.origin[0]) / self.resolution)
        row = int((wy - self.origin[1]) / self.resolution)
        return col, row

    def _is_occupied(self, col, row):
        """障害物セルかどうか（範囲外は壁扱い）"""
        h, w = self.grid.shape
        if row < 0 or row >= h or col < 0 or col >= w:
            return True
        return self.grid[row, col] != 0

    def _cast_ray(self, fx, fy, dx, dy):
        """1本のレイを grid 解像度ステップでマーチングして障害物を検出する。

        Parameters
        ----------
        fx, fy : float  出発点（ワールド座標）
        dx, dy : float  方向単位ベクトル

        Returns
        -------
        (distance, hit_x, hit_y)
        """
        step = self.resolution          # 1 ステップ = 1 grid セル幅
        max_steps = int(self.MAX_DIST / step)

        for s in range(1, max_steps + 1):
            d = s * step
            # レイ方向にresolutionステップずつ進めた連続ワールド座標がwx,wy
            wx = fx + dx * d
            wy = fy + dy * d
            col, row = self._world_to_cell(wx, wy)
            if self._is_occupied(col, row):
                return d, wx, wy

        d = self.MAX_DIST
        return d, fx + dx * d, fy + dy * d

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def scan(self, odom_x, odom_y, odom_yaw):
        """オドメトリ自己位置からグリッドマップ上でスキャンを行う。

        Parameters
        ----------
        odom_x   : float  オドメトリ推定 X 座標 [m]
        odom_y   : float  オドメトリ推定 Y 座標 [m]
        odom_yaw : float  オドメトリ推定 yaw 角 [rad]

        Returns
        -------
        distances : list[float]        各レイの検出距離 [m]
        ray_from  : list[[x, y, z]]    レイ出発点リスト
        ray_to    : list[[x, y, z]]    レイ終端点リスト
        """
        # robot.urdf lidar_joint: ロボット前方 lidar_offset_x への固定オフセット
        lx = odom_x + math.cos(odom_yaw) * self.lidar_offset_x
        ly = odom_y + math.sin(odom_yaw) * self.lidar_offset_x

        distances = []
        ray_from  = []
        ray_to    = []

        for i in range(self.NUM_RAYS):
            angle = odom_yaw + (-self.FOV / 2 + self.FOV * i / (self.NUM_RAYS - 1))
            dx = math.cos(angle)
            dy = math.sin(angle)

            dist, hx, hy = self._cast_ray(lx, ly, dx, dy)

            distances.append(dist)
            ray_from.append([lx, ly, self.lidar_height])
            ray_to.append([hx, hy, self.lidar_height])

        return distances, ray_from, ray_to
