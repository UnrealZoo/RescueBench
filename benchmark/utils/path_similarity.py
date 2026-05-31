"""轨迹相似度计算器 — DTW / Fréchet / Hausdorff"""

import numpy as np


class PathSimilarityCalculator:
    """轨迹相似度计算器 (DTW, Fréchet, Hausdorff)"""

    @staticmethod
    def dtw_distance(traj1: np.ndarray, traj2: np.ndarray, use_3d: bool = True) -> float:
        """Dynamic Time Warping 距离 (归一化)"""
        if use_3d:
            traj1 = traj1[:, :3] if traj1.shape[1] > 3 else traj1
            traj2 = traj2[:, :3] if traj2.shape[1] > 3 else traj2

        n, m = len(traj1), len(traj2)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(traj1[i - 1] - traj2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m] / max(n, m)

    @staticmethod
    def frechet_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Fréchet 距离 (考虑轨迹顺序性)"""
        traj1 = traj1[:, :3] if traj1.shape[1] > 3 else traj1
        traj2 = traj2[:, :3] if traj2.shape[1] > 3 else traj2

        n, m = len(traj1), len(traj2)
        ca = np.full((n, m), -1.0)

        def _c(i: int, j: int) -> float:
            if ca[i, j] > -0.5:
                return ca[i, j]
            d = np.linalg.norm(traj1[i] - traj2[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(_c(i - 1, 0), d)
            elif i == 0 and j > 0:
                ca[i, j] = max(_c(0, j - 1), d)
            else:
                ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
            return ca[i, j]

        return _c(n - 1, m - 1)

    @staticmethod
    def hausdorff_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Hausdorff 距离 (最大最小距离)"""
        traj1 = traj1[:, :3] if traj1.shape[1] > 3 else traj1
        traj2 = traj2[:, :3] if traj2.shape[1] > 3 else traj2

        def directed(a, b):
            return max(min(np.linalg.norm(a[i] - b[j]) for j in range(len(b)))
                       for i in range(len(a)))

        return max(directed(traj1, traj2), directed(traj2, traj1))

    @staticmethod
    def normalized_similarity(distance: float, max_distance: float = 5000.0) -> float:
        """距离 → 0-1 相似度 (1.0 = 完全相同)"""
        return max(0.0, 1.0 - distance / max_distance)
