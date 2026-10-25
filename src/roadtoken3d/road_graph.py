from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class RoadProjection:
    road_index: int
    external_road_id: int
    s_road: float
    center_xyz: np.ndarray
    heading_rad: float
    grade: float
    signed_lateral_m: float
    horizontal_distance_m: float

class RoadGraph:

    def __init__(self, directory: str | Path, mmap_mode: str | None='r') -> None:
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text('utf-8'))
        names = ('points_xyz', 's_road', 'headings', 'grades', 'point_road_index', 'road_ptr', 'road_external_id', 'road_length_m', 'road_width_m', 'road_mean_z', 'road_mean_grade', 'road_sin_heading', 'road_cos_heading', 'adjacency_ptr', 'adjacency_index', 'grid_order', 'grid_codes_sorted')
        for name in names:
            setattr(self, name, np.load(self.directory / f'{name}.npy', mmap_mode=mmap_mode))
        counts = np.diff(self.road_ptr).astype(np.float64)
        self.road_mean_x = (np.add.reduceat(np.asarray(self.points_xyz[:, 0], dtype=np.float64), self.road_ptr[:-1]) / counts).astype(np.float32)
        self.road_mean_y = (np.add.reduceat(np.asarray(self.points_xyz[:, 1], dtype=np.float64), self.road_ptr[:-1]) / counts).astype(np.float32)
        self.feature_xy_center = np.asarray([self.road_mean_x.mean(), self.road_mean_y.mean()], dtype=np.float32)
        self.feature_xy_scale = float(max(self.road_mean_x.std(), self.road_mean_y.std(), 1.0))
        transition_ptr = self.directory / 'transition_ptr.npy'
        transition_index = self.directory / 'transition_index.npy'
        if transition_ptr.exists() and transition_index.exists():
            self.transition_ptr = np.load(transition_ptr, mmap_mode=mmap_mode)
            self.transition_index = np.load(transition_index, mmap_mode=mmap_mode)
        else:
            self.transition_ptr = None
            self.transition_index = None
        runtime_features = self.directory / 'runtime_node_features.npy'
        cached_features = np.load(runtime_features, mmap_mode=mmap_mode) if runtime_features.exists() else None
        self._node_feature_cache = cached_features if cached_features is not None and cached_features.ndim == 2 and (cached_features.shape == (self.num_roads, 8)) else None
        self._candidate_table_cache: dict[int, np.ndarray] = {}

    @property
    def num_roads(self) -> int:
        return int(len(self.road_external_id))

    def neighbors(self, road_index: int) -> np.ndarray:
        left = int(self.adjacency_ptr[road_index])
        right = int(self.adjacency_ptr[road_index + 1])
        return np.asarray(self.adjacency_index[left:right], dtype=np.int32)

    def transition_allowed(self, previous: int, current: int) -> bool:
        if previous == current:
            return True
        return bool(np.any(self.neighbors(previous) == current))

    def candidate_roads(self, road_index: int, maximum: int=16) -> np.ndarray:
        pieces = [np.asarray([road_index], dtype=np.int32), self.neighbors(road_index)]
        if self.transition_ptr is not None and self.transition_index is not None:
            left = int(self.transition_ptr[road_index])
            right = int(self.transition_ptr[road_index + 1])
            pieces.append(np.asarray(self.transition_index[left:right], dtype=np.int32))
        values = np.concatenate(pieces)
        (_, first) = np.unique(values, return_index=True)
        return values[np.sort(first)][:maximum]

    def transition_targets(self, road_index: int) -> np.ndarray:
        """Return train-only observed next roads, if preprocessing produced them."""
        if self.transition_ptr is None or self.transition_index is None:
            return np.empty(0, dtype=np.int32)
        left = int(self.transition_ptr[road_index])
        right = int(self.transition_ptr[road_index + 1])
        return np.asarray(self.transition_index[left:right], dtype=np.int32)

    def node_features(self, road_indices: np.ndarray) -> np.ndarray:
        index = np.asarray(road_indices, dtype=np.int64)
        if self._node_feature_cache is not None:
            return np.asarray(self._node_feature_cache[index], dtype=np.float32)
        return np.stack((np.asarray(self.road_mean_z[index]) / 1000.0, np.asarray(self.road_mean_grade[index]), np.asarray(self.road_sin_heading[index]), np.asarray(self.road_cos_heading[index]), np.log1p(np.asarray(self.road_length_m[index])) / 10.0, np.asarray(self.road_width_m[index]) / 20.0, (np.asarray(self.road_mean_x[index]) - self.feature_xy_center[0]) / self.feature_xy_scale, (np.asarray(self.road_mean_y[index]) - self.feature_xy_center[1]) / self.feature_xy_scale), axis=-1).astype(np.float32)

    def runtime_candidate_table(self, maximum: int, require_cache: bool=False) -> tuple[np.ndarray, np.ndarray]:
        """Return a dense road-neighbor lookup used by the training Dataset.

        The published Ascend image includes this table, avoiding hundreds of
        Python/NumPy graph operations for every trajectory sample.
        """
        maximum = int(maximum)
        if maximum in self._candidate_table_cache:
            values = self._candidate_table_cache[maximum]
            return (values, values >= 0)
        path = self.directory / f'runtime_candidate_indices_{maximum}.npy'
        if path.exists():
            values = np.load(path, mmap_mode='r')
        else:
            if require_cache:
                raise RuntimeError(f'Ascend runtime cache is missing: {path}. Run scripts/prepare_ascend_runtime_cache.py before building the image.')
            values = np.full((self.num_roads, maximum), -1, dtype=np.int32)
            for road_index in range(self.num_roads):
                candidates = self.candidate_roads(road_index, maximum)
                values[road_index, :len(candidates)] = candidates
        self._candidate_table_cache[maximum] = values
        return (values, values >= 0)

    def write_runtime_cache(self, maximum: int=16) -> dict[str, int]:
        """Materialize small dense lookup arrays outside the NPU job."""
        feature_path = self.directory / 'runtime_node_features.npy'
        if self._node_feature_cache is None:
            indices = np.arange(self.num_roads, dtype=np.int64)
            features = self.node_features(indices)
            temporary = feature_path.with_suffix('.tmp.npy')
            np.save(temporary, features, allow_pickle=False)
            temporary.replace(feature_path)
            self._node_feature_cache = np.load(feature_path, mmap_mode='r')
        candidate_path = self.directory / f'runtime_candidate_indices_{int(maximum)}.npy'
        if not candidate_path.exists():
            (values, _) = self.runtime_candidate_table(int(maximum), require_cache=False)
            np.save(candidate_path, np.asarray(values, dtype=np.int32), allow_pickle=False)
            self._candidate_table_cache.pop(int(maximum), None)
        (values, valid) = self.runtime_candidate_table(int(maximum), require_cache=True)
        return {'roads': self.num_roads, 'neighbors': int(valid.sum()), 'maximum_neighbors': int(maximum)}

    def _grid_code(self, cell_x: int, cell_y: int) -> int:
        return (cell_x - int(self.metadata['grid_min_x'])) * int(self.metadata['grid_ny']) + cell_y - int(self.metadata['grid_min_y'])

    def nearby_point_indices(self, x: float, y: float, radius_m: float) -> np.ndarray:
        cell_size = float(self.metadata['grid_cell_m'])
        qx = int(math.floor(x / cell_size))
        qy = int(math.floor(y / cell_size))
        cell_radius = int(math.ceil(radius_m / cell_size))
        pieces: list[np.ndarray] = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                code = self._grid_code(qx + dx, qy + dy)
                left = int(np.searchsorted(self.grid_codes_sorted, code, side='left'))
                right = int(np.searchsorted(self.grid_codes_sorted, code, side='right'))
                if right > left:
                    pieces.append(np.asarray(self.grid_order[left:right], dtype=np.int32))
        if not pieces:
            return np.empty(0, dtype=np.int32)
        candidates = np.concatenate(pieces)
        points = np.asarray(self.points_xyz[candidates, :2])
        keep = np.sum((points - np.asarray([x, y])) ** 2, axis=1) <= radius_m ** 2
        return candidates[keep]

    def nearby_road_indices(self, x: float, y: float, radius_m: float, maximum: int) -> np.ndarray:
        """Return unique nearby roads ordered by nearest sampled centerline point."""
        point_indices = self.nearby_point_indices(x, y, radius_m)
        if not len(point_indices):
            return np.empty(0, dtype=np.int32)
        points = np.asarray(self.points_xyz[point_indices, :2])
        distance = np.sum((points - np.asarray([x, y])) ** 2, axis=1)
        ranked_roads = np.asarray(self.point_road_index[point_indices[np.argsort(distance)]])
        (_, first) = np.unique(ranked_roads, return_index=True)
        return ranked_roads[np.sort(first)][:maximum].astype(np.int32, copy=False)

    def _project_segment(self, point_xy: np.ndarray, left: int, right: int) -> RoadProjection:
        a = np.asarray(self.points_xyz[left], dtype=np.float64)
        b = np.asarray(self.points_xyz[right], dtype=np.float64)
        vector = b[:2] - a[:2]
        length_squared = float(np.dot(vector, vector))
        fraction = 0.0 if length_squared < 1e-09 else float(np.clip(np.dot(point_xy - a[:2], vector) / length_squared, 0.0, 1.0))
        center = a + fraction * (b - a)
        road_index = int(self.point_road_index[left])
        s_value = float(self.s_road[left] + fraction * (self.s_road[right] - self.s_road[left]))
        heading = math.atan2(float(vector[1]), float(vector[0]))
        tangent = vector / max(math.sqrt(length_squared), 1e-09)
        relative = point_xy - center[:2]
        signed_lateral = float(tangent[0] * relative[1] - tangent[1] * relative[0])
        return RoadProjection(road_index=road_index, external_road_id=int(self.road_external_id[road_index]), s_road=s_value, center_xyz=center.astype(np.float32), heading_rad=heading, grade=float(self.grades[left] + fraction * (self.grades[right] - self.grades[left])), signed_lateral_m=signed_lateral, horizontal_distance_m=float(np.linalg.norm(relative)))

    def project_candidates(self, point_xy: np.ndarray, radius_m: float=150.0, maximum_roads: int=8) -> list[RoadProjection]:
        point_xy = np.asarray(point_xy, dtype=np.float64)
        point_indices = self.nearby_point_indices(float(point_xy[0]), float(point_xy[1]), radius_m)
        if not len(point_indices):
            return []
        distances = np.sum((np.asarray(self.points_xyz[point_indices, :2]) - point_xy[None, :]) ** 2, axis=1)
        ranked = point_indices[np.argsort(distances)[:max(32, maximum_roads * 6)]]
        best_by_road: dict[int, RoadProjection] = {}
        for point_index in ranked:
            road_index = int(self.point_road_index[point_index])
            start = int(self.road_ptr[road_index])
            stop = int(self.road_ptr[road_index + 1])
            segments: list[tuple[int, int]] = []
            if int(point_index) > start:
                segments.append((int(point_index) - 1, int(point_index)))
            if int(point_index) + 1 < stop:
                segments.append((int(point_index), int(point_index) + 1))
            if not segments:
                projection = RoadProjection(road_index=road_index, external_road_id=int(self.road_external_id[road_index]), s_road=float(self.s_road[point_index]), center_xyz=np.asarray(self.points_xyz[point_index], dtype=np.float32), heading_rad=float(self.headings[point_index]), grade=float(self.grades[point_index]), signed_lateral_m=0.0, horizontal_distance_m=float(np.linalg.norm(point_xy - np.asarray(self.points_xyz[point_index, :2], dtype=np.float64))))
                candidates = [projection]
            else:
                candidates = [self._project_segment(point_xy, left, right) for (left, right) in segments]
            for projection in candidates:
                previous = best_by_road.get(road_index)
                if previous is None or projection.horizontal_distance_m < previous.horizontal_distance_m:
                    best_by_road[road_index] = projection
        return sorted(best_by_road.values(), key=lambda value: value.horizontal_distance_m)[:maximum_roads]

    def project_to_road(self, road_index: int, point_xy: np.ndarray) -> RoadProjection:
        """Project a point to one selected road; used by constrained decoding."""
        point_xy = np.asarray(point_xy, dtype=np.float64)
        start = int(self.road_ptr[road_index])
        stop = int(self.road_ptr[road_index + 1])
        if stop - start == 1:
            point = np.asarray(self.points_xyz[start], dtype=np.float32)
            return RoadProjection(road_index, int(self.road_external_id[road_index]), float(self.s_road[start]), point, float(self.headings[start]), float(self.grades[start]), 0.0, float(np.linalg.norm(point_xy - point[:2])))
        points = np.asarray(self.points_xyz[start:stop, :2], dtype=np.float64)
        nearest = int(np.argmin(np.sum((points - point_xy) ** 2, axis=1))) + start
        candidates: list[RoadProjection] = []
        if nearest > start:
            candidates.append(self._project_segment(point_xy, nearest - 1, nearest))
        if nearest + 1 < stop:
            candidates.append(self._project_segment(point_xy, nearest, nearest + 1))
        return min(candidates, key=lambda value: value.horizontal_distance_m)

    def point_at_s(self, road_index: int, s_value: float) -> RoadProjection:
        start = int(self.road_ptr[road_index])
        stop = int(self.road_ptr[road_index + 1])
        values = np.asarray(self.s_road[start:stop])
        right_local = int(np.searchsorted(values, s_value, side='right'))
        right_local = min(max(right_local, 1), max(1, len(values) - 1))
        if len(values) == 1:
            point = np.asarray(self.points_xyz[start], dtype=np.float32)
            return RoadProjection(road_index, int(self.road_external_id[road_index]), float(values[0]), point, float(self.headings[start]), float(self.grades[start]), 0.0, 0.0)
        left = start + right_local - 1
        right = start + right_local
        left_s = float(self.s_road[left])
        right_s = float(self.s_road[right])
        fraction = 0.0 if right_s <= left_s else float(np.clip((s_value - left_s) / (right_s - left_s), 0.0, 1.0))
        a = np.asarray(self.points_xyz[left], dtype=np.float64)
        b = np.asarray(self.points_xyz[right], dtype=np.float64)
        center = a + fraction * (b - a)
        vector = b[:2] - a[:2]
        heading = math.atan2(float(vector[1]), float(vector[0]))
        grade = float(self.grades[left] + fraction * (self.grades[right] - self.grades[left]))
        return RoadProjection(road_index=road_index, external_road_id=int(self.road_external_id[road_index]), s_road=left_s + fraction * (right_s - left_s), center_xyz=center.astype(np.float32), heading_rad=heading, grade=grade, signed_lateral_m=0.0, horizontal_distance_m=0.0)

    def is_multilevel(self, x: float, y: float, road_index: int, road_z: float, horizontal_radius_m: float=30.0, vertical_separation_m: float=8.0) -> bool:
        indices = self.nearby_point_indices(x, y, horizontal_radius_m)
        if not len(indices):
            return False
        other_road = np.asarray(self.point_road_index[indices]) != int(road_index)
        vertical = np.abs(np.asarray(self.points_xyz[indices, 2]) - float(road_z))
        return bool(np.any(other_road & (vertical >= vertical_separation_m)))
