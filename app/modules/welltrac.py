from config.importation import *
from modules.wellpad import WPad
from modules.wellzone import WZone

class WTrack (WPad):
    def __init__(self, variable_file):
        WPad.__init__(self, variable_file)
        self._pf_dict_wt_trajectories = None
        self._pf_dict_wt_lenghts_vs = None
        self._pf_dict_wt_lenghts_gs = None
        self._pf_dict_wt_lenghts_ds = None
        self._pf_dict_wt_wp_names = None
        self._pf_dict_wt_quantity_ds = None
        self._pf_dict_wt_values = None
        self._pf_dict_wt_zone_points = None
        self._pf_dict_wt_zone_weight = None
        self._dict_wt_trajectorie = None
        self._dict_wt_lenght_vs = None
        self._dict_wt_lenght_gs = None
        self._dict_wt_lenght_ds = None
        self._dict_wt_wp_names = None
        self._dict_wt_quantity_ds = None
        self._dict_wt_values = None
        self._dict_wt_ds_names = None
        self._dict_wtds_md_start = None
        self._dict_wtds_md_end = None

    class Wellaccount:
        @staticmethod
        def sort_line(arr: np.ndarray, point: np.ndarray, index: bool = False) -> np.ndarray:
            """
            Сортировка массива точек arr относительно point на близость \n
            :param arr: Координаты точек траектории;
            :param point: Устьевая координаты;
            :return: Отсортированный массив
            """
            dist_matrix = distance_matrix(arr[:, :2], arr[:, :2])
            pairs = np.unravel_index(np.argsort(dist_matrix, axis=None)[::-1], dist_matrix.shape)
            pairs = np.array(pairs).T
            max_angle = np.inf
            closest_idx_, farthest_idx_ = None, None
            for pair in pairs:
                p1_idx, p2_idx = pair
                if p1_idx == p2_idx:
                    continue
                dist_to_point1 = distance_matrix(arr[:, :2][p1_idx].reshape(1, -1), point[:2].reshape(1, -1))[0][0]
                dist_to_point2 = distance_matrix(arr[:, :2][p2_idx].reshape(1, -1), point[:2].reshape(1, -1))[0][0]
                if dist_to_point1 < dist_to_point2:
                    closest_idx, farthest_idx = p2_idx, p1_idx
                else:
                    closest_idx, farthest_idx = p1_idx, p2_idx
                v1 = arr[:, :2][closest_idx] - arr[:, :2][farthest_idx]
                v2 = point[:2] - arr[:, :2][farthest_idx]
                angle = WTrack.Wellaccount.calculate_angle(v1, v2)
                if angle >= 90:
                    closest_idx_ = closest_idx
                    farthest_idx_ = farthest_idx
                    break
                elif angle > max_angle:
                    max_angle = angle
                    closest_idx_ = closest_idx
                    farthest_idx_ = farthest_idx
            if closest_idx_ is None:
                if not index:
                    return arr
                else:
                    return np.arange(len(arr))
            remaining_indices = np.setdiff1d(np.arange(len(arr)), [closest_idx_, farthest_idx_])
            distances_to_closest = distance_matrix(arr, arr[closest_idx_].reshape(1, -1))[:, 0]
            sorted_indices = np.argsort(distances_to_closest)
            max_dist_indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            distances_to_farthest = distance_matrix(arr[remaining_indices], arr[farthest_idx_].reshape(1, -1))[:, 0]
            sorted_remaining_indices = remaining_indices[np.argsort(distances_to_farthest)]
            sorted_indices = np.concatenate(([farthest_idx_], sorted_remaining_indices, [closest_idx_]))
            if not index:
                return arr[sorted_indices]
            else:
                return sorted_indices

        @staticmethod
        def approximate(points: np.ndarray) -> np.ndarray:
            """
            Полиноминальная апроксимация 2 степени \n
            :param points: Исходный набор точек
            :return: Скорректированный набор точек
            """
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            degree = 3
            coeffs_y = np.polyfit(x, y, degree)
            coeffs_z = np.polyfit(x, z, degree)
            poly_y = np.poly1d(coeffs_y)
            poly_z = np.poly1d(coeffs_z)
            x_new = np.linspace(x[0], x[-1], num=100)
            y_new = poly_y(x_new)
            z_new = poly_z(x_new)
            new_points = np.column_stack((x_new, y_new, z_new))
            return new_points

        @staticmethod
        def smoothing_path(points):
            """
            Полиноминальная интерполяция 2 степени \n
            :param points: Исходный набор точек
            :return: Скорректированный набор точек
            """
            tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0, k=2)
            u_new = np.linspace(0, 1, 100)
            new_points = np.array(splev(u_new, tck)).T
            return new_points

        @staticmethod
        def compute_curvature(points: np.ndarray) -> float:
            """
            Расчет кривизны траектории \n
            :param points: Исходный набор точек
            :return: Значение кривизны
            """
            dx_dt = np.gradient(points[:, 0])
            dy_dt = np.gradient(points[:, 1])
            dz_dt = np.gradient(points[:, 2])
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)
            d2z_dt2 = np.gradient(dz_dt)
            curvature = np.abs(d2x_dt2 + d2y_dt2 + d2z_dt2) / (np.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2) ** 3)
            return np.cumsum(curvature)[-1]

        @staticmethod
        def calculate_lenght(points: np.ndarray, option: bool = False) -> float or List:
            """
            Расчет длины траектории \n
            :param points: Исходный набор точек
            :param option: Опция записи MD
            :return: Значение длины
            """
            if not option:
                length = 0.0
                for i in range(1, points.shape[0]):
                    length += euclidean(points[i - 1], points[i])
                return length
            else:
                lengths = np.array([0])
                for i in range(1, len(points)):
                    segment_length = euclidean(points[i - 1], points[i])
                    lengths = np.append(lengths, lengths[-1] + segment_length)
                return lengths

        @staticmethod
        def interpolate_path(path_points: np.ndarray, num_points: int = 10, option: bool = True):
            """
            Интерполирует путь для добавления дополнительных точек между каждой парой точек.
            :param path_points: Список точек пути
            :param num_points: Общее количество точек в интерполированном пути
            :param option: Возможность отаботки точек с дубликатами
            :return: Интерполированный путь в формате np.array()
            """
            # Интерполируем сплайнами
            if option:
                tck, u = splprep(path_points.T, s=0, k=1)
                u_new = np.linspace(0, 1, num_points)
                new_points = splev(u_new, tck)
                return np.array(new_points).T
            else:
                seen = set()
                unique_points = []
                for point in path_points:
                    point_tuple = tuple(point)
                    if point_tuple not in seen:
                        seen.add(point_tuple)
                        unique_points.append(point)
                unique_points = np.array(unique_points)
                if len(unique_points) == 1:
                    return False, None
                elif len(unique_points) < len(path_points):
                    tck, u = splprep(unique_points.T, s=0, k=1)
                    u_new = np.linspace(0, 1, num_points)
                    new_points = splev(u_new, tck)
                    return True, np.array(new_points).T
                else:
                    tck, u = splprep(unique_points.T, s=0, k=1)
                    u_new = np.linspace(0, 1, num_points)
                    new_points = splev(u_new, tck)
                    return True, np.array(new_points).T

        @staticmethod
        def hierarchical_tree(points: np.ndarray, n: int = 10) -> List:
            """
            Кластеризует точки с использованием K-means, строит графы с кратчайшими путями для каждого кластера
            и объединяет их в один граф.
            :param arr: Массив точек в формате np.array([[x1, y1], [x2, y2], ...])
            :param n: Процент создания кластеров (целое число от 0 до 100)
            :return: Объединенный граф, содержащий все графы с кратчайшими путями
            """
            first_point = points[0]
            points = points[1:]
            num_clusters = int(points.shape[0] * n / 100)
            if num_clusters <= 1:
                mst = WTrack.Graph.dijkstra_tree(points)
                paths = []
                for node in mst.nodes():
                    path = nx.shortest_path(mst, source=node, target=0)
                    if len(path) != 2:
                        continue
                    path_length = nx.shortest_path_length(mst, source=node, target=0)
                    total_length = path_length + euclidean(first_point, points[node])
                    path_points = np.array([first_point, points[path[1]], points[path[0]]])
                    paths.append((path_points, total_length))
                return [path for path, _ in sorted(paths, key=lambda x: x[1], reverse=True)]
            cluster_labels, cluster_centers = WZone.Cluster.create_simple_kmeans(points, values=None, n=num_clusters, option=True)
            clusters = {i: [cluster_centers[i]] for i in range(num_clusters)}
            for i, label in enumerate(cluster_labels):
                clusters[label].append(points[i])
            paths = []
            for i, cluster in clusters.items():
                mst = WTrack.Graph.dijkstra_tree(np.array(cluster))
                for node in mst.nodes():
                    path = nx.shortest_path(mst, source=node, target=0)
                    if len(path) != 2:
                        continue
                    path_length = nx.shortest_path_length(mst, source=node, target=0)
                    total_length = path_length + euclidean(first_point, cluster[node])
                    path_points = np.array([first_point, cluster[path[1]], cluster[path[0]]])
                    paths.append((path_points, total_length))
            return [path for path, _ in sorted(paths, key=lambda x: x[1], reverse=True)]

        @staticmethod
        def spanning_tree(points: np.ndarray, n: int = 10) -> nx.Graph:
            """
            Кластеризует точки с использованием K-means, строит минимальные остовные деревья для каждого кластера
            и объединяет их в один граф.
            :param points: Массив точек в формате np.array([[x1, y1, z1], [x2, y2, z2], ...])
            :param n: Процент создания кластеров (целое число от 0 до 100)
            :return: Объединенный граф, содержащий все минимальные остовные деревья
            """
            # Исключаем первую точку
            first_point = points[0]
            points = points[1:]
            num_clusters = int(points.shape[0] * n / 100)
            if num_clusters <= 1:
                points = np.vstack([first_point, points])
                mst = WTrack.Graph.prim_tree(points)
                paths = []
                for node in mst.nodes():
                    path = nx.shortest_path(mst, source=node, target=0)
                    path_length = nx.shortest_path_length(mst, source=node, target=0)
                    total_length = path_length
                    path_points = np.array([points[i] for i in path])
                    paths.append((path_points, total_length))
                return [path for path, _ in sorted(paths, key=lambda x: x[1], reverse=True)]
            cluster_labels, cluster_centers = WZone.Cluster.create_simple_kmeans(points, values=None, n=num_clusters, option=True)
            cluster_centers = np.vstack([first_point, cluster_centers])
            mst_centers = WTrack.Graph.prim_tree(cluster_centers)
            clusters = {i: [cluster_centers[i]] for i in range(num_clusters + 1)}
            for i, label in enumerate(cluster_labels):
                clusters[label + 1].append(points[i])
            paths = []
            for index, node in enumerate(mst_centers.nodes()):
                if node == 0:
                    continue
                center_path = nx.shortest_path(mst_centers, source=0, target=node)
                center_path_length = nx.shortest_path_length(mst_centers, source=0, target=node)
                center_point = cluster_centers[node]
                cluster_points = clusters[node]
                mst_cluster = WTrack.Graph.prim_tree(np.array(cluster_points))
                for sub_node in mst_cluster.nodes():
                    if sub_node == 0:
                        continue
                    try:
                        sub_path = nx.shortest_path(mst_cluster, source=sub_node, target=0)
                        sub_path_length = nx.shortest_path_length(mst_cluster, source=sub_node, target=0)
                        total_length = center_path_length + sub_path_length
                        path_points = np.array([cluster_centers[i] for i in center_path] + [cluster_points[i] for i in sub_path])
                        paths.append((path_points, total_length))
                    except nx.NetworkXNoPath:
                        continue
            return [path for path, _ in sorted(paths, key=lambda x: x[1], reverse=True)]

        @staticmethod
        def nearest_tree(points: np.ndarray, main_line: np.ndarray, param: int) -> np.ndarray:
            results = []
            P0, P1 = main_line
            P0 = P0.reshape(3, 1)
            P1 = P1.reshape(3, 1)
            direction = P1 - P0
            for point in points:
                point = point.reshape(3, 1)
                P0_to_point = point - P0
                t = np.dot(P0_to_point.T, direction) / np.dot(direction.T, direction)
                intersection_point = P0 + t * direction
                intersection_to_point = point - intersection_point
                normalized_intersection_to_point = intersection_to_point / np.linalg.norm(intersection_to_point)
                adjusted_intersection_point = intersection_point + (param / 100.0) * np.linalg.norm(
                    intersection_to_point) * normalized_intersection_to_point
                results.append(
                    (P0.flatten(), (adjusted_intersection_point.flatten() + P0.flatten()) / 2, point.flatten()))
            return np.array(results)

        @staticmethod
        def calculate_angle(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
            """
            Рассчитывает угол между двумя векторами.
            """
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(cos_angle))
            return angle

        @staticmethod
        def chek_inresection(arr_1: np.ndarray, general: List[np.ndarray]) -> bool:
            interpolated_path_2d = arr_1[:, :2]
            trajectories_2d = [tra[:, :2] for tra in general]
            interpolated_line = LineString(interpolated_path_2d)
            trajectory_lines = [LineString(tra) for tra in trajectories_2d]
            intersection_points = []
            for tra_line in trajectory_lines:
                intersection = interpolated_line.intersection(tra_line)
                if not intersection.is_empty:
                    if intersection.geom_type == 'Point':
                        intersection_points.append(np.array(intersection.coords[0]))
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            intersection_points.append(np.array(point.coords[0]))
            if not intersection_points:
                return False
            if len(intersection_points) != 1:
                return False
            last_point = interpolated_path_2d[0]
            all_points = np.vstack(trajectories_2d)
            dist_matrix = distance_matrix(last_point.reshape(1, -1), all_points)
            closest_point_index = np.argmin(dist_matrix)
            closest_point = all_points[closest_point_index]
            if not np.all(np.isclose(closest_point, last_point)):
                print('Ближайшая точка != arr_1[0]')
                return False

            return True

        @staticmethod
        def chek_angle_waste(arr_1: np.ndarray, general: List[np.ndarray]) -> bool:
            point_t = general[0][-1]
            point_m = arr_1[-1]
            point_e = arr_1[0]
            v1 = point_t - point_m
            v2 = point_e - point_m
            angle = WTrack.Wellaccount.calculate_angle(v1, v2)
            if angle <= 90:
                return False
            else:
                return True

    class Wiring:
        def __init__(self, target_step_distance: float = 10, target_angle_intensity: float = 2):
            self.target_step_distance = target_step_distance
            self.target_angle_intensity = target_angle_intensity

        @property
        def target_curvature(self):
            """
            Рассчитывает кривизну дуги окружности, описанной вокруг многоугольника.
            Параметры:
            side_length (float): Длина стороны многоугольника.
            angle_degrees (float): Угол между сторонами многоугольника в градусах.
            Возвращает:
            float: Кривизна дуги окружности.
            """
            angle_radians = np.radians(self.target_angle_intensity)
            p0 = np.array([0, 0, 0])
            p1 = np.array([self.target_step_distance, 0, 0])
            p2 = np.array(
                [self.target_step_distance * np.cos(angle_radians),
                 self.target_step_distance * np.sin(angle_radians), 0])
            dx = np.gradient([p0[0], p1[0], p2[0]], edge_order=2)
            dy = np.gradient([p0[1], p1[1], p2[1]], edge_order=2)
            dz = np.gradient([p0[2], p1[2], p2[2]], edge_order=2)
            ddx = np.gradient(dx, edge_order=2)
            ddy = np.gradient(dy, edge_order=2)
            ddz = np.gradient(dz, edge_order=2)
            velocity = np.vstack((dx, dy, dz)).T
            acceleration = np.vstack((ddx, ddy, ddz)).T
            cross_product = np.cross(velocity, acceleration)
            norm_velocity = np.linalg.norm(velocity, axis=1)
            curvature = np.linalg.norm(cross_product, axis=1) / norm_velocity ** 3
            return np.max(curvature)

        @staticmethod
        def partition_target(points: np.ndarray, step_distance: float = 10) -> np.ndarray:
            """
            Интерполирует массив точек так, чтобы расстояние между соседними точками было равно step_distance.
            Параметры:
            points (np.array): Массив точек с shape = (N, 3), где N >= 2.
            step_distance (float): Желаемое расстояние между соседними точками.
            Возвращает:
            np.array: Интерполированный массив точек с shape = (M, 3), где M >= N.
            """
            segments = np.diff(points, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            cumulative_lengths = np.cumsum(lengths)
            cumulative_lengths = np.insert(cumulative_lengths, 0, 0)
            new_lengths = np.arange(0, cumulative_lengths[-1], step_distance)
            new_points = np.zeros((len(new_lengths), 3))
            for i in range(3):
                interpolator = interp1d(cumulative_lengths, points[:, i], kind='linear')
                new_points[:, i] = interpolator(new_lengths)
            if np.abs(new_lengths[-1] - cumulative_lengths[-1]) > 1e-6:
                new_points = np.vstack((new_points, points[-1]))
            return new_points

        @staticmethod
        def target_interset_point(line_gs: np.ndarray, line_ds: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
            """
            Находит точку пересечения двух линий в плоскости XY.
            :param line_gs: np.array shape(N, 3). N>=2
            :param line_ds: np.array shape(N, 3). N>=2
            :return: Точка пересечения.
            """
            x1, y1 = line_gs[-2][:2]
            x2, y2 = line_gs[-1][:2]
            x3, y3 = line_ds[0][:2]
            x4, y4 = line_ds[-1][:2]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return False, None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            intersection_point = np.array([px, py, (line_gs[-1][2] + line_ds[0][2]) / 2])
            if not (min(line_gs[-1][0], line_ds[0][0]) <= px <= max(line_gs[-1][0], line_ds[0][0]) and
                    min(line_gs[-1][1], line_ds[0][1]) <= py <= max(line_gs[-1][1], line_ds[0][1])):
                return -1, intersection_point
            return 1, intersection_point

        @staticmethod
        def target_index_distance(point: np.ndarray, points: np.ndarray, R: float):
            distances = np.linalg.norm(points - point, axis=1)
            exact_match_index = np.where(distances == R)[0]
            if len(exact_match_index) > 0:
                return exact_match_index[0]
            closest_index = np.argmin(np.abs(distances - R))
            return closest_index

        @staticmethod
        def create_bezier_curve_3_point(p0, p1, p2, num_points=10):
            """
            Создает квадратичную кривую Безье между точками p0 и p2 с контрольной точкой p1.
            Параметры:
            p0 (np.array): Начальная точка.
            p1 (np.array): Контрольная точка.
            p2 (np.array): Конечная точка.
            num_points (int): Количество точек на кривой.
            Возвращает:
            np.array: Массив точек на кривой Безье.
            """
            t = np.linspace(0, 1, num_points)
            curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, p1) + np.outer(t ** 2, p2)
            return curve

        @staticmethod
        def calculate_bezier_curvature(curve_line, p_1, p_2, step_distance=10):
            """
            Рассчитывает кривизну кривой Безье.
            Параметры:
            curve_line (np.array): Кривая Безье.
            p_1 (np.array): Начальная точка.
            p_2 (np.array): Конечная точка.
            step_distance (float): Геометрический шаг между соседними точками.
            Возвращает:
            float: Максимальная кривизна кривой Безье.
            """
            mod_curve = np.vstack((p_1[0], curve_line, p_2[1]))
            new_curve = WTrack.Wiring.partition_target(mod_curve, step_distance)
            dx = np.gradient(new_curve[:, 0], edge_order=2)
            dy = np.gradient(new_curve[:, 1], edge_order=2)
            dz = np.gradient(new_curve[:, 2], edge_order=2)
            ddx = np.gradient(dx, edge_order=2)
            ddy = np.gradient(dy, edge_order=2)
            ddz = np.gradient(dz, edge_order=2)
            velocity = np.vstack((dx, dy, dz)).T
            acceleration = np.vstack((ddx, ddy, ddz)).T
            cross_product = np.cross(velocity, acceleration)
            norm_velocity = np.linalg.norm(velocity, axis=1)
            curvature = np.linalg.norm(cross_product, axis=1) / norm_velocity ** 3
            return np.max(curvature)

        @staticmethod
        def create_bezier_curve_4_point(p0, p1, p2, p3, num_points=10):
            """
            Создает квадратичную кривую Безье между точками p0 и p2 с контрольной точкой p1.
            Параметры:
            p0 (np.array): Начальная точка.
            p1 (np.array): Контрольная точка.
            p2 (np.array): Контрольная точка.
            p3 (np.array): Конечная точка.
            num_points (int): Количество точек на кривой.
            Возвращает:
            np.array: Массив точек на кривой Безье.
            """
            t = np.linspace(0, 1, num_points)
            curve = (
                    np.outer((1 - t) ** 3, p0) +
                    np.outer(3 * (1 - t) ** 2 * t, p1) +
                    np.outer(3 * (1 - t) * t ** 2, p2) +
                    np.outer(t ** 3, p3)
            )
            return curve

        @staticmethod
        def calculate_bezier_curvature(curve_line, p_1, p_2, step_distance=10):
            """
            Рассчитывает кривизну кривой Безье.
            Параметры:
            curve_line (np.array): Кривая Безье.
            p_1 (np.array): Начальная точка.
            p_2 (np.array): Конечная точка.
            step_distance (float): Геометрический шаг между соседними точками.
            Возвращает:
            float: Максимальная кривизна кривой Безье.
            """
            try:
                mod_curve = np.vstack((p_1[0], curve_line, p_2[1]))
                new_curve = WTrack.Wiring.partition_target(mod_curve, step_distance)
                dx = np.gradient(new_curve[:, 0], edge_order=2)
                dy = np.gradient(new_curve[:, 1], edge_order=2)
                dz = np.gradient(new_curve[:, 2], edge_order=2)
                ddx = np.gradient(dx, edge_order=2)
                ddy = np.gradient(dy, edge_order=2)
                ddz = np.gradient(dz, edge_order=2)
                velocity = np.vstack((dx, dy, dz)).T
                acceleration = np.vstack((ddx, ddy, ddz)).T
                cross_product = np.cross(velocity, acceleration)
                norm_velocity = np.linalg.norm(velocity, axis=1)
                curvature = np.zeros_like(norm_velocity)
                valid_indices = norm_velocity > 1e-8
                curvature[valid_indices] = np.linalg.norm(cross_product[valid_indices], axis=1) / norm_velocity[
                    valid_indices] ** 3
                curvature = np.linalg.norm(cross_product, axis=1) / norm_velocity ** 3
                return np.max(curvature)
            except ValueError:
                return float('inf')

        def cutting_mode(self, line_gs: np.ndarray, line_ds: np.ndarray) -> Optional[np.ndarray]:
            """
            Опция для определения оптимальных конфигруаии срезок ДС
            Args:
                line_gs: Коордианты доп. ствола
                line_ds: Коордианты ГС
            Returns: ДС
            """
            R = 300
            status, intersection_point = WTrack.Wiring.target_interset_point(line_gs, line_ds)
            if status is None:
                return None
            distances = np.linalg.norm(line_gs[:, :2] - intersection_point[:2], axis=1)
            closest_index = np.argmin(distances)
            if closest_index == 0:
                return None
            index_gs_start = WTrack.Wiring.target_index_distance(line_gs[closest_index], line_gs[:closest_index], R)
            index_ds_end = WTrack.Wiring.target_index_distance(line_ds[0], line_ds, R)
            num_samples = 20
            p0_values = np.linspace(line_gs[index_gs_start], line_gs[closest_index], num_samples)
            p2_values = np.linspace(line_ds[0], line_ds[index_ds_end], num_samples)
            best_curvature = float('inf')
            best_curvature_ = float('inf')
            best_p0 = None
            best_p2 = None
            for i, p0 in enumerate(p0_values):
                for j, p2 in enumerate(p2_values):
                    curve = WTrack.Wiring.create_bezier_curve_3_point(p0, intersection_point, p2)
                    curvature = WTrack.Wiring.calculate_bezier_curvature(curve, line_gs, line_ds)
                    if abs(curvature - self.target_curvature) < abs(best_curvature - self.target_curvature):
                        best_curvature = curvature
                        best_p0 = p0
                        best_p2 = p2
            if best_p0 is not None and best_p2 is not None:
                curve = WTrack.Wiring.create_bezier_curve_3_point(best_p0, intersection_point, best_p2)
                remaining_line_ds = np.vstack((best_p2, line_ds[-1]))
                final_curve = np.vstack((curve, remaining_line_ds))
                return final_curve
            else:
                return None

        def transition_trajectory_mode(self, line_vs: np.ndarray, line_gs: np.ndarray) -> Optional[np.ndarray]:
                """
                Опция для определения оптимальных конфигурации вертикальной и горизонтальной части
                Args:
                    line_gs: Координаты доп. ствола
                    line_ds: Координаты ГС
                    R1: Расстояние удлинения line_ds
                    R2: Расстояние сокращения line_gs
                Returns: ДС
                """
                num_point = 10
                R1_interval = np.linspace(0, np.linalg.norm(line_vs[-1] - line_vs[0]) - 500 - 100, num_point)
                R2_interval = np.linspace(500, 5000, num_point)
                direction_vs = (line_vs[-1] - line_vs[0]) / np.linalg.norm(line_vs[-1] - line_vs[0])
                direction_gs = (line_gs[0] - line_gs[-1]) / np.linalg.norm(line_gs[0] - line_gs[-1])
                p1_interval = [line_vs[-1] - i * direction_vs for i in R1_interval]
                p2_interval = [line_gs[0] + i * direction_gs for i in R2_interval]
                best_curvature = float('inf')
                best_p1 = None
                best_p2 = None
                p0 = line_vs[0]
                p3 = line_gs[0]
                mid_point = np.array([p0[0], p0[1], 500])
                for p1, r1 in zip(p1_interval, R1_interval):
                    for p2, r2 in zip(p2_interval, R2_interval):
                        curve = WTrack.Wiring.create_bezier_curve_4_point(mid_point, p1, p2, p3)
                        curvature = WTrack.Wiring.calculate_bezier_curvature(curve, line_vs, line_gs)
                        if abs(curvature - self.target_curvature) < abs(best_curvature - self.target_curvature):
                            best_curvature = curvature
                            best_p1 = p1
                            best_p2 = p2
                # Если не было найдено подходящей кривизны, ищем минимальную кривизну
                if best_p1 is None or best_p2 is None:
                    best_curvature = float('inf')
                    for p1, r1 in zip(p1_interval, R1_interval):
                        for p2, r2 in zip(p2_interval, R2_interval):
                            curve = WTrack.Wiring.create_bezier_curve_4_point(mid_point, p1, p2, p3)
                            curvature = WTrack.Wiring.calculate_bezier_curvature(curve, line_vs, line_gs)
                            if curvature <= best_curvature:
                                best_curvature = curvature
                                best_p1 = p1
                                best_p2 = p2
                if best_p1 is not None and best_p2 is not None:
                    curve = WTrack.Wiring.create_bezier_curve_4_point(mid_point, best_p1, best_p2, p3)
                    final_curve = np.vstack((p0, curve))
                    return final_curve
                else:
                    return None

    class Graph:
        @staticmethod
        def kruskal_tree(points: np.ndarray) -> nx.Graph:
            """
            Функция для нахождения минимального остовного дерева с использованием алгоритма Крускала.
            :param points: Набор точек
            :return: Минимальное остовное дерево в виде графа NetworkX
            """
            dist_matrix = squareform(pdist(points, metric='euclidean'))
            G = nx.from_numpy_array(dist_matrix)
            mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
            return mst

        @staticmethod
        def prim_tree(points: np.ndarray) -> nx.Graph:
            """
            Функция для нахождения минимального остовного дерева с использованием алгоритма Прима.
            :param points: Набор точек
            :return: Минимальное остовное дерево в виде графа NetworkX
            """
            dist_matrix = squareform(pdist(points, metric='euclidean'))
            G = nx.from_numpy_array(dist_matrix)
            mst = nx.minimum_spanning_tree(G, algorithm='prim')
            return mst

        @staticmethod
        def boruvka_tree(points: np.ndarray) -> nx.Graph:
            """
            Функция для нахождения минимального остовного дерева с использованием алгоритма Борувки.
            :param points: Набор точек
            :return: Минимальное остовное дерево в виде графа NetworkX
            """
            dist_matrix = squareform(pdist(points, metric='euclidean'))
            G = nx.from_numpy_array(dist_matrix)
            mst = nx.minimum_spanning_tree(G, algorithm='boruvka')
            return mst

        @staticmethod
        def dijkstra_tree(points: np.ndarray, source: int = 0) -> nx.Graph:
            """
            Функция для нахождения дерева кратчайших путей с использованием алгоритма Дейкстры.
            :param points: Набор точек
            :param source: Исходная вершина
            :return: Дерево кратчайших путей в виде графа NetworkX
            """
            dist_matrix = squareform(pdist(points, metric='euclidean'))
            G = nx.from_numpy_array(dist_matrix)
            shortest_paths = nx.single_source_dijkstra_path(G, source)
            path_graph = nx.Graph()
            for target, path in shortest_paths.items():
                for i in range(len(path) - 1):
                    path_graph.add_edge(path[i], path[i + 1])
            return path_graph

        @staticmethod
        def bellman_ford_tree(points: np.ndarray, source: int = 0) -> nx.Graph:
            """
            Функция для нахождения дерева кратчайших путей с использованием алгоритма Беллмана-Форда.
            :param points: Набор точек
            :param source: Исходная вершина
            :return: Дерево кратчайших путей в виде графа NetworkX
            """
            dist_matrix = squareform(pdist(points, metric='euclidean'))
            G = nx.from_numpy_array(dist_matrix)
            shortest_paths = nx.single_source_bellman_ford_path(G, source)
            path_graph = nx.Graph()
            for target, path in shortest_paths.items():
                for i in range(len(path) - 1):
                    path_graph.add_edge(path[i], path[i + 1])
            return path_graph

    class Emissions:
        @staticmethod
        def outliers_if(points, contamination=0.5):
            """
            Удаляет выбросы из набора точек, используя алгоритм Isolation Forest.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            contamination (float): Доля выбросов в данных.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")

            # Применяем Isolation Forest для определения выбросов
            clf = IsolationForest(contamination=contamination)
            clf.fit(points)
            inlier_indices = np.where(clf.predict(points) == 1)[0]
            return inlier_indices

        @staticmethod
        def outliers_ee(points, contamination=0.5):
            """
            Удаляет выбросы из набора точек, используя алгоритм Elliptic Envelope.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            contamination (float): Доля выбросов в данных.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            clf = EllipticEnvelope(contamination=contamination)
            clf.fit(points)
            inlier_indices = np.where(clf.predict(points) == 1)[0]
            return inlier_indices

        @staticmethod
        def outliers_lof(points, contamination=0.5):
            """
            Удаляет выбросы из набора точек, используя алгоритм Local Outlier Factor (LOF).
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            contamination (float): Доля выбросов в данных.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            clf = LocalOutlierFactor(contamination=contamination)
            outliers = clf.fit_predict(points)
            inlier_indices = np.where(outliers == 1)[0]
            return inlier_indices

        @staticmethod
        def outliers_optics(points, min_samples=5, xi=0.05):
            """
            Удаляет выбросы из набора точек, используя алгоритм OPTICS.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            min_samples (int): Минимальное количество точек для формирования кластера.
            xi (float): Параметр для определения скорости роста плотности.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            clust = OPTICS(min_samples=min_samples, xi=xi)
            clust.fit(points)
            inlier_indices = np.where(clust.labels_ != -1)[0]
            return inlier_indices

        @staticmethod
        def outliers_zscore(points, threshold=1):
            """
            Удаляет выбросы из набора точек, используя метод Z-оценок.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            threshold (float): Пороговое значение Z-оценки для определения выбросов.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            z_scores = np.abs((points - mean) / std)
            inlier_indices = np.where(np.all(z_scores < threshold, axis=1))[0]
            return inlier_indices

        @staticmethod
        def outliers_bootstrap(points, confidence_level=0.95):
            """
            Удаляет выбросы из набора точек, используя метод Bootstrap.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            confidence_level (float): Уровень доверия для доверительных интервалов.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            lower_bounds = []
            upper_bounds = []
            for i in range(points.shape[1]):
                data = points[:, i]
                res = bootstrap((data,), np.mean, confidence_level=confidence_level)
                lower_bounds.append(res.confidence_interval.low)
                upper_bounds.append(res.confidence_interval.high)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            inlier_indices = np.where(
                np.all(points >= lower_bounds, axis=1) & np.all(points <= upper_bounds, axis=1)
            )[0]
            return inlier_indices

        @staticmethod
        def remove_outliers_kde(points, bandwidth=0.5, threshold=0.1):
            """
            Удаляет выбросы из набора точек, используя метод ядерной оценки плотности (KDE).
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            bandwidth (float): Ширина полосы пропускания для KDE.
            threshold (float): Пороговое значение плотности для определения выбросов.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(points)
            log_densities = kde.score_samples(points)
            densities = np.exp(log_densities)
            inlier_indices = np.where(densities > threshold)[0]
            return inlier_indices

        @staticmethod
        def outliers_meanshift(points, bandwidth=None):
            """
            Удаляет выбросы из набора точек, используя алгоритм Mean Shift.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            bandwidth (float): Ширина полосы пропускания для Mean Shift. Если None, будет вычислена автоматически.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(points)
            inlier_indices = np.where(ms.labels_ != -1)[0]
            return inlier_indices

        @staticmethod
        def outliers_svm(points, kernel='rbf', gamma='scale', nu=0.05):
            """
            Удаляет выбросы из набора точек, используя алгоритм One-Class SVM.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            kernel (str): Тип ядра для SVM.
            gamma (str or float): Коэффициент для ядра.
            nu (float): Параметр, контролирующий долю выбросов.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
            svm.fit(points)
            inlier_indices = np.where(svm.predict(points) == 1)[0]
            return inlier_indices

        @staticmethod
        def outliers_mcd(points, support_fraction=0.9):
            """
            Удаляет выбросы из набора точек, используя алгоритм Minimum Covariance Determinant (MCD).
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            support_fraction (float): Доля точек, используемых для оценки ковариационной матрицы.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            mcd = MinCovDet(support_fraction=support_fraction)
            mcd.fit(points)
            distances = mcd.mahalanobis(points)
            threshold = np.percentile(distances, 95)
            inlier_indices = np.where(distances <= threshold)[0]
            return inlier_indices

        @staticmethod
        def outliers_contour(points, threshold=0.3):
            """
            Удаляет выбросы из набора точек, используя контурный анализ данных.
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            threshold (float): Пороговое значение для определения выбросов.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]
            distances = np.min(np.linalg.norm(points[:, np.newaxis] - hull_vertices, axis=2), axis=1)
            inlier_indices = np.where(distances <= threshold)[0]
            return inlier_indices

        @staticmethod
        def outliers_gmm(points, n_components=100, covariance_type='full', threshold=2.5):
            """
            Удаляет выбросы из набора точек, используя Gaussian Mixture Models (GMM).
            Параметры:
            points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            n_components (int): Количество компонентов (кластеров) для GMM.
            covariance_type (str): Тип ковариации для GMM.
            threshold (float): Пороговое значение для определения выбросов.
            Возвращает:
            numpy.ndarray: Массив индексов точек, которые не являются выбросами.
            """
            if points.shape[1] != 3:
                raise ValueError("Массив точек должен иметь форму (n, 3)")
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            gmm.fit(points)
            log_prob = gmm.score_samples(points)
            inlier_indices = np.where(log_prob > np.percentile(log_prob, threshold))[0]
            return inlier_indices

        @staticmethod
        def points_inside(contur_points, points):
            """
            Определяет краевые точки contur_points, строит по ним многоугольник и находит точки points, которые находятся внутри многоугольника.
            Параметры:
            contur_points (numpy.ndarray): Массив точек с формой (n, 3), где n - количество точек.
            points (numpy.ndarray): Массив точек с формой (m, 3), где m - количество точек.
            Возвращает:
            numpy.ndarray: Массив индексов точек points, которые находятся внутри многоугольника.
            """
            contur_points_2D = contur_points[:, :2]
            points_2D = points[:, :2]
            hull = ConvexHull(contur_points_2D)
            vertices = contur_points_2D[hull.vertices]
            polygon = Polygon(vertices)
            inside_indices = np.array([i for i, point in enumerate(points_2D) if polygon.contains(Point(point))])
            return inside_indices

    def __create_wells(self):
        for wp_current, wp in enumerate(self.get_wp_names):
            self.logger.log_info(f"Well pad {wp} (x; y):  {self.get_dict_a_wp_points[wp]}, zones:  {self.get_dict_wp_wz_names[wp]}")
        mode = WTrack.Wiring(self.wt_section_lenght, self.wt_section_angle)
        dict_wt_trajectories = {}
        dict_wt_lenght_vs = {}
        dict_wt_lenght_gs = {}
        dict_wt_lenght_ds = {}
        dict_wt_quantity_ds = {}
        dict_wt_wp_names = {}
        dict_wt_values = {}
        dict_wt_zone_weight = {}
        dict_wt_zone_points = {}
        algoritm = 'nearest'
        for wp_current, wp in enumerate(self.get_wp_names):
            for wz_current, wz in enumerate(self.get_dict_wp_wz_names[wp]):
                err = 0
                err_angle = 0
                err_dist = 0
                err_cutting = 0
                err_trac_index = 0
                err_insert = 0
                err_intersekt = 0

                """ Подбор конфигурации траектории  """

                trac = go.Figure()
                wp_point_2d = self.get_dict_a_wp_points[wp]
                # Продумать момент с определением альтитуды
                wp_point_3d = np.append(wp_point_2d, 0)
                if self.result_save:
                    trac.add_trace(go.Scatter3d(
                        x=[wp_point_3d[0]],
                        y=[wp_point_3d[1]],
                        z=[wp_point_3d[2]],
                        name='Устье',
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='red'
                        )
                    ))
                wz_points = self.get_dict_a_belong_points[wz]
                wz_values = self.get_dict_wz_values[wz]
                if self.result_save:
                    trac.add_trace(go.Scatter3d(
                        x=wz_points[:, 0],
                        y=wz_points[:, 1],
                        z=wz_points[:, 2],
                        name='Точки с выбросами',
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=wz_values,
                            opacity=0.4,
                            colorscale='Jet',
                        )
                    ))
                try:
                    emissions_points = wz_points[WTrack.Emissions.outliers_mcd(wz_points)]
                    emissions_points = wz_points[WTrack.Emissions.points_inside(emissions_points, wz_points)]
                except Exception:
                    self.logger.log_info(f'Zone {wz} excluded due to small number of source points')
                    continue
                if len(emissions_points) <= self.well_threshold:
                    self.logger.log_info(f'Zone {wz} excluded due to small number of source points')
                    continue
                emissions_weights = self.get_current_values(emissions_points)
                indices = np.where(emissions_weights >= np.median(emissions_weights))[0]
                emissions_points = emissions_points[indices]
                emissions_weights = emissions_weights[indices]
                dict_wt_zone_points[wz] = emissions_points
                dict_wt_zone_weight[wz] = emissions_weights
                if self.result_save:
                    trac.add_trace(go.Scatter3d(
                        x=emissions_points[:, 0],
                        y=emissions_points[:, 1],
                        z=emissions_points[:, 2],
                        name='Точки без выбросов',
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=emissions_weights,
                            colorscale='Inferno',
                            opacity=0.4,
                        )
                    ))
                _, cluster_points, cluster_weights = WZone.Cluster.create_simple_kmeans(emissions_points, emissions_weights, emissions_points.shape[0] // 5, True)
                track_index = WTrack.Wellaccount.sort_line(cluster_points, wp_point_3d, index=True)
                track_points = cluster_points[track_index]
                track_weights = cluster_weights[track_index]
                selected_points, selected_weights = np.array([track_points[0], track_points[-1]]), np.array(
                    [track_weights[0], track_weights[-1]])
                for _ in range(self.truc_threshold):
                    distance_matrix = cdist(track_points, selected_points)
                    min_distances = np.min(distance_matrix, axis=1)
                    max_distance_index = np.argmax(min_distances)
                    selected_points = np.vstack((selected_points, track_points[max_distance_index]))
                    selected_weights = np.append(selected_weights, track_weights[max_distance_index])
                if self.result_save:
                    trac.add_trace(go.Scatter3d(
                        x=selected_points[:, 0],
                        y=selected_points[:, 1],
                        z=selected_points[:, 2],
                        name='Опорные точки',
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=selected_weights,
                            colorscale='Inferno',
                            opacity=0.2,
                        )
                    ))
                if algoritm == 'nearest':
                    # Построение дерева через перпендикуляры
                    main_line = np.vstack((track_points[0], track_points[-1]))
                    mst = WTrack.Wellaccount.nearest_tree(selected_points[2:], main_line, self.graph_vertex)
                else:
                    # Построение минимального остовного дерева
                    mst = WTrack.Wellaccount.hierarchical_tree(selected_points, n=self.graph_vertex)
                    if len(mst) == 0:
                        continue
                interp_paths = []
                for path in mst:
                    status, ipath =WTrack.Wellaccount.interpolate_path(path, option=False)
                    if status:
                        interp_paths.append(ipath)
                if self.result_save:
                    for i, path in enumerate(interp_paths):
                        trac.add_trace(go.Scatter3d(
                            x=path[:, 0],
                            y=path[:, 1],
                            z=path[:, 2],
                            name=f'ГЛ-{i + 1}',
                            mode='lines',
                            line=dict(
                                color='grey',
                            )
                        ))
                smoothed_paths = []
                for path in interp_paths:
                    try:
                        smoothed_paths.append(WTrack.Wellaccount.approximate(path))
                    except Exception:
                        continue
                if self.result_save:
                    for i, path in enumerate(smoothed_paths):
                        trac.add_trace(go.Scatter3d(
                            x=path[:, 0],
                            y=path[:, 1],
                            z=path[:, 2],
                            name=f'АЛ-{i + 1}',
                            mode='lines',
                            line=dict(
                                color='orange',
                            )
                        ))
                smoothed_paths.sort(key=lambda path: -np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                if algoritm == 'nearest':
                    if np.linalg.norm(main_line[-1] - main_line[0]) > self.length_max:
                        gs_lenght = np.linalg.norm(main_line[-1] - main_line[0])
                        gs_line_morm = (main_line[-1] - main_line[0]) / gs_lenght
                        main_line[-1] = main_line[0] + self.length_max * gs_line_morm
                    trajectories = [WTrack.Wiring.partition_target(main_line)]
                else:
                    for path in smoothed_paths:
                        distances_to_start = np.linalg.norm(path - path[0], axis=1)
                        distances_to_end = np.linalg.norm(path - path[-1], axis=1)
                        total_distances = distances_to_start + distances_to_end
                        mid_point = path[np.argmax(total_distances)]
                        v1 = path[-1] - mid_point
                        v2 = path[0] - mid_point
                        angle = WTrack.Wellaccount.calculate_angle(v1, v2)
                        if angle <= 170:
                            continue
                        v1 = np.vstack((path[0], path[-1]))
                        v1 = WTrack.Wiring.partition_target(v1)
                        trajectories = [v1]
                        break
                smoothed_paths = [i[int(i.shape[0] * (self.trimed_vertex / 100)):] for i in smoothed_paths]
                step = 0
                ires = 0
                ires_status = False
                while True:
                    if step == self.wt_max_iter:
                        break
                    step += 1
                    if ires_status:
                        ires += 1
                        ires_status = False
                        if ires >= self.wt_const_iter:
                            break
                    else:
                        ires = 0
                    if len(trajectories) >= self.wt_max:
                        break
                    for status, path in enumerate(smoothed_paths):
                        err += 1
                        path = np.vstack((path[0], path[-1]))
                        path = WTrack.Wiring.partition_target(path)
                        min_dist = np.inf
                        trac_intersection_line_index = None
                        for trajectory_index, trajectory in enumerate(trajectories):
                            if not self.stvol_in_stvol:
                                if trajectory_index != 0:
                                    continue
                            distances = cdist([path[0]], trajectory).flatten()
                            closest_index = np.argmin(distances)
                            dist = distances[closest_index]
                            if dist < min_dist:
                                min_dist = dist
                                trac_intersection_line_index = trajectory_index
                        if self.result_save:
                            trac.add_trace(go.Scatter3d(
                                x=path[:, 0],
                                y=path[:, 1],
                                z=path[:, 2],
                                name=f'path-{status}',
                                mode='lines',
                                line=dict(
                                    color='purple',
                                )
                            ))
                        if not self.stvol_in_stvol:
                            if trac_intersection_line_index != 0:
                                err_trac_index += 1
                                continue
                        status_target = False
                        iteration = 50
                        v1 = path[-1] - path[0]
                        v2 = trajectories[trac_intersection_line_index][-1] - trajectories[trac_intersection_line_index][0]
                        angle = WTrack.Wellaccount.calculate_angle(v1, v2)
                        corr_path = mode.cutting_mode(trajectories[trac_intersection_line_index], path)
                        if self.angle_min <= angle <= self.angle_max and corr_path is not None:
                            status_target = True
                        else:
                            min_target = int(len(trajectories[trac_intersection_line_index]) * 0.3)
                            max_target = int(len(trajectories[trac_intersection_line_index]) * 0.75)
                            target = (min_target + max_target) // 2
                            while True:
                                if iteration <= 0:
                                    break
                                iteration -= 1
                                v1 = path[-1] - trajectories[trac_intersection_line_index][target]
                                angle = WTrack.Wellaccount.calculate_angle(v1, v2)
                                if self.angle_min <= angle <= self.angle_max:
                                    status_target = True
                                    path = np.vstack((path[-1], trajectories[trac_intersection_line_index][target]))
                                    path = WTrack.Wiring.partition_target(path)
                                    corr_path = mode.cutting_mode(trajectories[trac_intersection_line_index], path)
                                    if corr_path is not None:
                                        status_target = True
                                        break
                                if angle > self.angle_max:
                                    target -= 1
                                elif angle < self.angle_min:
                                    target += 1
                                if target > max_target or target < min_target:
                                    break
                        if not status_target:
                            err_insert += 1
                            continue
                        if corr_path is None:
                            err_cutting += 1
                            continue
                        if self.result_save:
                            trac.add_trace(go.Scatter3d(
                                x=[path[-1][0]],
                                y=[path[-1][1]],
                                z=[path[-1][2]],
                                name=f'Начальная точка ds - {status}',
                                mode='markers+text',
                                marker=dict(
                                    color='red',
                                ),
                                text=[status, angle],
                                textposition="top center"
                            ))
                            trac.add_trace(go.Scatter3d(
                                x=[path[0][0]],
                                y=[path[0][1]],
                                z=[path[0][2]],
                                name=f'Конечная точка ds - {status}',
                                mode='markers+text',
                                marker=dict(
                                    color='blue',
                                ),
                                text=[status, angle],
                                textposition="top center"
                            ))
                        interpolated_path = WTrack.Wiring.partition_target(corr_path)
                        path_length = WTrack.Wellaccount.calculate_lenght(interpolated_path)
                        if self.length_min >= path_length or self.length_max <= path_length:
                            err_dist += 1
                            continue
                        min_distance_to_trajectories = min(
                            euclidean(interpolated_path[-1], point) for trajectory in trajectories for point in
                            trajectory)
                        if min_distance_to_trajectories < self.distance:
                            err_dist += 1
                            continue
                        if not WTrack.Wellaccount.chek_angle_waste(interpolated_path, trajectories):
                            err_angle += 1
                            continue
                        distances = np.sqrt(np.sum((trajectories[0] - interpolated_path[0]) ** 2, axis=1))
                        nearest_point_index = np.argmin(distances)
                        nearest_point = trajectories[0][nearest_point_index]
                        interpolated_path = np.insert(interpolated_path, 0, nearest_point, axis=0)
                        interpolated_path = WTrack.Wiring.partition_target(interpolated_path, 10)
                        if not WTrack.Wellaccount.chek_inresection(interpolated_path, trajectories):
                            err_intersekt += 1
                            continue
                        trajectories.append(interpolated_path)
                        ires_status = True
                self.logger.log_info(f"Well - | {wz} | WP - {wp} | completion configuration selected")

                ''' Идентификация траекторий скважин '''

                ds = 1
                fig = go.Figure()
                for index, curve_line in enumerate(trajectories):
                    if index == 0:
                        reference_points = np.array([
                            np.array(
                                [
                                    wp_point_3d[0],
                                    wp_point_3d[1] + wz_current * self.mouth_distance,
                                    wp_point_3d[2]
                                ]),
                            np.array(
                                [
                                    wp_point_3d[0],
                                    wp_point_3d[1] + wz_current * self.mouth_distance,
                                    800
                                ]),
                        ])
                        gs_points = np.array([
                            curve_line[0],
                            curve_line[round(curve_line.shape[0] * 0.5)]
                        ])
                        VS_line = mode.transition_trajectory_mode(reference_points, gs_points)
                        VS_line = WTrack.Wiring.partition_target(VS_line, 10)
                        if self.result_save:
                            trac.add_trace(go.Scatter3d(
                                x=VS_line[:, 0],
                                y=VS_line[:, 1],
                                z=VS_line[:, 2],
                                name='вертикальная часть',
                                mode='lines',
                                line=dict(
                                    color='blue',
                                )
                            ))
                        # Полное определение скважины только с ГС
                        GS_line = np.vstack((VS_line, curve_line[1:]))
                        # Добавляем ГС / Интервалы пефорации / Длину ГС
                        dict_wt_trajectories[str(wz)] = [VS_line, curve_line[1:]]
                        dict_wt_lenght_vs[str(wz)] = [WTrack.Wellaccount.calculate_lenght(VS_line)]
                        dict_wt_lenght_gs[str(wz)] = [WTrack.Wellaccount.calculate_lenght(curve_line[1:])]
                        dict_wt_quantity_ds[str(wz)] = 0
                        dict_wt_lenght_ds[str(wz)] = [0]
                        dict_wt_wp_names[str(wz)] = str(wp)
                        dict_wt_values[str(wz)] = [sum(self.get_current_values(curve_line))]
                        self.logger.log_info(f"Well - | {wz} | WP - {wp} | the main well trajectory has been calculated")
                    else:
                        if ds >= self.wt_max:
                            self.logger.log_info(f"Branch limit reached,  wt_max = {self.wt_max}")
                            continue
                        dict_wt_trajectories[str(wz)].append(curve_line)
                        dict_wt_lenght_ds[str(wz)].append(WTrack.Wellaccount.calculate_lenght(curve_line))
                        dict_wt_quantity_ds[str(wz)] = ds
                        dict_wt_values[str(wz)].append(sum(self.get_current_values(curve_line)))
                        self.logger.log_info(f"Well - | {wz} | WP - {wp} | additional barrel calculated | {ds} |")
                        ds += 1
                        continue
                self.logger.log_info(f'The combination of curves for {wz} wells at {wp} KP is calculated')
                if self.result_save:
                    for line in trajectories:
                        trac.add_trace(go.Scatter3d(
                            x=line[:, 0],
                            y=line[:, 1],
                            z=line[:, 2],
                            name='траектория',
                            mode='lines',
                            line=dict(
                                color='black',
                            )
                        ))
                    trac.update_layout(
                        scene=dict(
                            zaxis=dict(
                                range=[self.z_max + 200, 0]),
                            bgcolor='rgba(255,255,255,1)',  # Устанавливаем белый фон для сцены
                            aspectratio=dict(x=1, y=1, z=1)
                        )
                    )
                    trac.write_html(f"{self.dict_path_folders['trac_graph']}//trac_{wz}.html")
                    self.logger.log_info(f"Saving visualization of the well trajectory {wz}: {self.dict_path_folders['trac_graph']}//trac_{wz}.html")
                table = PrettyTable()
                table.field_names = ["NAME", "VALUE"]
                table.align["NAME"] = "l"
                table.align["VALUE"] = "r"
                table.border = True
                table.hrules = True
                table.vrules = True
                table.add_row(["Well Pad                                                  ", str(wp)])
                table.add_row(["Well name                                                 ", str(wz)])
                table.add_row(["Coordinates of the mouth                                  ", f'{round(wp_point_3d[0], 2)}, {round(wp_point_3d[1], 2)}'])
                table.add_row(["Altitude                                                m.", str(round(wp_point_3d[2], 2))])
                table.add_row(["Passage to T2,                                          m.", str(round(sum(dict_wt_lenght_vs[str(wz)]), 2))])
                table.add_row(["Passage of the horizontal part,                         m.", str(round(sum(dict_wt_lenght_gs[str(wz)]), 2))])
                table.add_row(["Sinking of side shafts,                                 m.", str(round(sum(dict_wt_lenght_ds[str(wz)]), 2))])
                table.add_row(["Number of side trunks,                                  ", str(dict_wt_quantity_ds[str(wz)])])
                table.add_row(["Total number of attempts to find side trunks,           ", str(err)])
                table.add_row(["Excluded due to corner offset,                          %", str(round(err_angle / err * 100))])
                table.add_row(["Excluded due to intersection(s),                        %", str(round(err_intersekt / err * 100))])
                table.add_row(["Excluded due to minimum distance,                       %", str(round(err_dist / err * 100))])
                table.add_row(["Excluded due to cutting angle,                          %", str(round(err_insert / err * 100))])
                table.add_row(["Excluded due to curvature,                              %", str(round(err_cutting / err * 100))])
                table.add_row(["Excluded due to barrel in barrel,                       %", str(round(err_trac_index / err * 100))])
                self.logger.log_info(f"\n{table}\n")
        self._pf_dict_wt_trajectories = dict_wt_trajectories
        self._pf_dict_wt_lenghts_vs = dict_wt_lenght_vs
        self._pf_dict_wt_lenghts_gs = dict_wt_lenght_gs
        self._pf_dict_wt_lenghts_ds = dict_wt_lenght_ds
        self._pf_dict_wt_wp_names = dict_wt_wp_names
        self._pf_dict_wt_quantity_ds = dict_wt_quantity_ds
        self._pf_dict_wt_values = dict_wt_values
        self._pf_dict_wt_zone_points = dict_wt_zone_points
        self._pf_dict_wt_zone_weight = dict_wt_zone_weight

    def __recreate_wells(self):
        filename = f"{self.dict_path_folders['current']}//welltrac.INC"
        if os.path.exists(filename):
            with open(filename, 'w') as f:
                pass
        else:
            with open(filename, 'w') as f:
                pass
        __dict_wt_trajectories = {}
        __dict_wt_lenght_vs = {}
        __dict_wt_lenght_gs = {}
        __dict_wt_lenght_ds = {}
        __dict_wt_wp_names = {}
        __dict_wt_quantity_ds = {}
        __dict_wt_values = {}
        __dict_wt_zone_points = {}
        __dict_wt_zone_weight = {}
        dict_wt_ds_names = {}
        dict_wtds_md_start = {}
        dict_wtds_md_end = {}
        _pf_dict_wp_wt_names = defaultdict(list)
        for w, wp in self._pf_dict_wt_wp_names.items():
            _pf_dict_wp_wt_names[wp].append(w)
        _pf_dict_wp_wt_names = dict(_pf_dict_wp_wt_names)
        for wp_current, wp in enumerate(_pf_dict_wp_wt_names.keys()):
            if len(_pf_dict_wp_wt_names[wp]) >= 2:
                for wt_current, wt in enumerate(_pf_dict_wp_wt_names[wp]):
                    for wt_index, curve_line in enumerate(self._pf_dict_wt_trajectories[wt]):
                        if wt_index == 0:
                            mod_line = curve_line
                        elif wt_index == 1:
                            with open(filename, 'a') as f:
                                f.write(f"welltrack '{wt}'\n")
                                dict_wt_ds_names[str(wt)] = [str(wt)]
                                dict_wtds_md_start[str(wt)] = round(sum(self._pf_dict_wt_lenghts_vs[wt]), 2)
                                dict_wtds_md_end[str(wt)] = round(sum(self._pf_dict_wt_lenghts_vs[wt]) + sum(self._pf_dict_wt_lenghts_gs[wt]) + 10, 2)
                                mod_line = np.vstack((mod_line, curve_line))
                                lengths = WTrack.Wellaccount.calculate_lenght(mod_line, option=True)
                                df = pd.DataFrame(mod_line, columns=['x', 'y', 'z'])
                                df['length'] = lengths
                                df = df.round({'x': 2, 'y': 2, 'z': 2, 'length': 2})
                                f.write(df.to_string(index=False, header=False, col_space=20, justify='right'))
                                f.write('\n/\n\n')
                        else:
                            with open(filename, 'a') as f:
                                start_length = 0
                                matching_point = None
                                matching_curve = None
                                for prev_curve in self._pf_dict_wt_trajectories[wt][:wt_index]:
                                    for p in prev_curve:
                                        if np.array_equal(curve_line[0], p):
                                            matching_point = p
                                            matching_curve = prev_curve
                                            break
                                    if matching_point is not None:
                                        break
                                if matching_point is not None:
                                    start_length = WTrack.Wellaccount.calculate_lenght(matching_curve, option=True)[-1]
                                f.write(f"welltrack '{wt}:{wt_index - 1}'\n")
                                lengths = WTrack.Wellaccount.calculate_lenght(curve_line, option=True)
                                lengths = [l + start_length for l in lengths]
                                df = pd.DataFrame(curve_line, columns=['x', 'y', 'z'])
                                df['length'] = lengths
                                df = df.round({'x': 2, 'y': 2, 'z': 2, 'length': 2})
                                df_str = df.to_string(index=False, header=False, col_space=20, justify='right')
                                f.write(df_str)
                                f.write('\n/\n\n')
                                dict_wt_ds_names[str(wt)].append(f"{wt}:{wt_index - 1}")
                                dict_wtds_md_start[f"{wt}:{wt_index - 1}"] = round(df['length'].iloc[0], 2)
                                dict_wtds_md_end[f"{wt}:{wt_index - 1}"] = round(df['length'].iloc[-1], 2)
                    __dict_wt_wp_names[wt] = wp
                    __dict_wt_trajectories[wt] = self._pf_dict_wt_trajectories[wt]
                    __dict_wt_lenght_vs[wt] = self._pf_dict_wt_lenghts_vs[wt]
                    __dict_wt_lenght_gs[wt] = self._pf_dict_wt_lenghts_gs[wt]
                    __dict_wt_lenght_ds[wt] = self._pf_dict_wt_lenghts_ds[wt]
                    __dict_wt_quantity_ds[wt] = self._pf_dict_wt_quantity_ds[wt]
                    __dict_wt_values[wt] = self._pf_dict_wt_values[wt]
                    __dict_wt_zone_points[wt] = self._pf_dict_wt_zone_points[wt]
                    __dict_wt_zone_weight[wt] = self._pf_dict_wt_zone_weight[wt]
            else:
                self.logger.log_warning(f"Well - | {_pf_dict_wp_wt_names[wp][0]} | WP - {wp} | excluded due to the limitation on the number of wells at the Well Pad")
        self._dict_wt_trajectories = __dict_wt_trajectories
        self._dict_wt_lenght_vs = __dict_wt_lenght_vs
        self._dict_wt_lenght_gs = __dict_wt_lenght_gs
        self._dict_wt_lenght_ds = __dict_wt_lenght_ds
        self._dict_wt_wp_names = __dict_wt_wp_names
        self._dict_wt_quantity_ds = __dict_wt_quantity_ds
        self._dict_wt_values = __dict_wt_values
        self._dict_wt_wz_points = __dict_wt_zone_points
        self._dict_wt_wz_weight = __dict_wt_zone_weight
        self._dict_wt_ds_names = dict_wt_ds_names
        self._dict_wtds_md_start = dict_wtds_md_start
        self._dict_wtds_md_end = dict_wtds_md_end
        del self._pf_dict_wt_zone_points
        del self._pf_dict_wt_zone_weight
        del self._pf_dict_wt_trajectories
        del self._pf_dict_wt_quantity_ds
        del self._pf_dict_wt_lenghts_vs
        del self._pf_dict_wt_lenghts_gs
        del self._pf_dict_wt_lenghts_ds
        del self._pf_dict_wt_values
        del self._pf_dict_wt_wp_names

    def __wt_visual(self):
        if self.result_save:
            fig = go.Figure()
            for well in self.get_dict_wt_trajectories.keys():
                for coord in self.get_dict_wt_trajectories[well]:
                    fig.add_trace(
                        go.Scatter3d(
                            x=coord[:, 0],
                            y=coord[:, 1],
                            z=coord[:, 2],
                            mode='lines',
                            line=dict(
                                width=6,
                                color='gray'
                            ),
                            name=f'well - {well}'
                        ))
            fig.add_trace(go.Scatter3d(
                x=self.get_ddnf_coords[:, 0],  # Координаты x
                y=self.get_ddnf_coords[:, 1],  # Координаты y
                z=self.get_ddnf_coords[:, 2],  # Координаты z
                mode='markers',
                marker=dict(
                    size=3,
                    opacity=0.1,
                    color=self.get_dfn_values
                )
            ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False, range=[self.x_min, self.x_max]),
                    yaxis=dict(visible=False, range=[self.y_min, self.y_max]),
                    zaxis=dict(visible=False, range=[self.z_max + 100, self.z_min - 100]),
                    bgcolor='rgba(255,255,255,1)',  # Устанавливаем белый фон для сцены
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                # showlegend=False,  # Убираем легенду
                margin=dict(l=0, r=0, b=0, t=0)  # Убираем отступы вокруг графика
            )
            fig.write_html(f"{self.dict_path_folders['current']}\wtrac.html")
            self.logger.log_info(f"Saving 3D Zone Cube{self.dict_path_folders['current']}\wtrac.html")

    def __call__(self, *args, **kwargs):
        WPad.__call__(self)
        self.__create_wells()
        self.__recreate_wells()
        self.__wt_visual()

    @property
    def get_dict_wt_trajectories(self) -> Dict[str, List[np.ndarray]]:
        """
        Свойство класса WTrack
        Returns: Координаты траектории
        """
        return self._dict_wt_trajectories

    @property
    def get_dict_wt_lenghts_vs(self) -> Dict[str, List[float]]:
        """
        Свойство класса WTrack
        Returns: Длина участка траектории - ветикальная часть скважины от устья до точки T2
        """
        return self._dict_wt_lenght_vs

    @property
    def get_dict_wt_lenghts_gs(self) -> Dict[str, List[float]]:
        """
        Свойство класса WTrack
        Returns: Длина участка траектории - горизонтальная часть - основной ствол
        """
        return self._dict_wt_lenght_gs

    @property
    def get_dict_wt_lenghts_ds(self) -> Dict[str, List[float]]:
        """
        Свойство класса WTrack
        Returns: Длина участка траектории - дополнительные стволы
        """
        return self._dict_wt_lenght_ds

    @property
    def get_dict_wt_quantity_ds(self) -> Dict[str, int]:
        """
        Свойство класса WTrack
        Returns: Количество дополнительных стволов
        """
        return self._dict_wt_quantity_ds

    @property
    def get_dict_wt_wp_names(self) -> Dict[str, str]:
        """
        Свойство класса WTrack
        Returns: Привязка скважина - куст
        """
        return self._dict_wt_wp_names

    @property
    def get_dict_wp_wt_names(self) -> Dict[str, str]:
        """
        Свойство класса WTrack
        Returns: Привязка куст - скважина
        """
        inverted_dict = defaultdict(list)
        for w, wp in self.get_dict_wt_wp_names.items():
            inverted_dict[wp].append(w)
        return dict(inverted_dict)

    @property
    def get_dict_wt_values(self):
        """
        Свойство класса WTrack
        Returns: Block_wells
        """
        return self._dict_wt_values

    @property
    def get_dict_wt_wz_points(self) -> Dict[str, np.ndarray]:
        return self._dict_wt_wz_points

    @property
    def get_dict_wt_wz_weight(self) -> Dict[str, np.ndarray]:
        return self._dict_wt_wz_weight

    @property
    def get_dict_wt_ds_names(self) -> Dict[str, List[str]]:
        return self._dict_wt_ds_names

    @property
    def get_dict_wtds_md_start(self) -> Dict[str, float]:
        return self._dict_wtds_md_start

    @property
    def get_dict_wtds_md_end(self) -> Dict[str, float]:
        return self._dict_wtds_md_end
