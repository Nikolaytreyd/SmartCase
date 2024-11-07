from modules.wellzone import WZone
from modules.transform import Transform
from config.importation import *

class WPad(WZone):

    def __init__(self, variable_file):
        WZone.__init__(self, variable_file)
        self.map_x_line = None
        self.map_y_line = None
        self.binary_map = None
        self._n_wp_points = None
        self._a_wp_points = None
        self._a_wp_wz_points = None
        self._n_wp_wz_points = None
        self._wp_values = None
        self._wp_wz_values = None
        self._wp_names = None
        self._wp_wz_names = None
        self._dict_n_wp_points = None
        self._dict_a_wp_points = None
        self._dict_n_wp_wz_points = None
        self._dict_a_wp_wz_points = None
        self._dict_wp_values = None
        self._dict_wp_wz_values = None
        self._dict_wp_wz_names = None

    class MapPoint:
        @staticmethod
        def points_map_corr(map: np.ndarray, map_line: tuple, points: np.ndarray, RR: float, R: float, R_check: float) -> np.ndarray:
            __normalized_map = map / 255
            __normalized_map = np.where(__normalized_map == 1, -1, __normalized_map)
            __normalized_map = np.where(__normalized_map == 0, 1, __normalized_map)
            __normalized_map = __normalized_map.flatten()
            __X, __Y = np.meshgrid(map_line[0], map_line[1])
            __ones_coords = np.column_stack((__X.flatten(), __Y.flatten()))
            __kd_tree = cKDTree(__ones_coords)
            corr_points = np.array([])
            for __point in points:
                __neighbor_indices_RR = __kd_tree.query_ball_point(__point, r=RR, p=2)
                __distances_RR = np.array(
                    [np.linalg.norm(__ones_coords[idx] - __point) for idx in __neighbor_indices_RR])
                __sorted_indices_RR = np.argsort(__distances_RR)
                __neighbor_indices_RR = np.array(__neighbor_indices_RR)[__sorted_indices_RR]
                __max_ones = -np.inf
                corr_point = None
                for __index in __neighbor_indices_RR:
                    __neighbor_coord = __ones_coords[__index]
                    __dist = distance.euclidean(__point, (__neighbor_coord[0], __neighbor_coord[1]))
                    if __dist <= RR:
                        __neighbor_indices_R = __kd_tree.query_ball_point(__neighbor_coord, r=R, p=2)
                        __zero_indices = __kd_tree.query_ball_point(__neighbor_coord, r=R_check, p=2)
                        if np.any(__normalized_map[__zero_indices] == 0):
                            __count = 0
                        else:
                            __weights = np.array([1 - (distance.euclidean((__neighbor_coord[0], __neighbor_coord[1]),
                                                                          __ones_coords[__i]) / R) for __i in
                                                  __neighbor_indices_R])
                            __weights /= np.sum(__weights)
                            __count = sum(__normalized_map[__neighbor_indices_R] * __weights)
                        if __count > __max_ones:
                            __max_ones = __count
                            corr_point = tuple(__neighbor_coord)
                if corr_point is None:
                    corr_point = __point
                corr_points = np.append(corr_points, corr_point)
                return corr_points.reshape(-1, 2)

        @staticmethod
        def point_map_corr(map: np.ndarray, map_line: tuple, point: tuple, RR: float, R: float,
                           R_check: float) -> tuple:
            __normalized_map = map / 255
            __normalized_map = np.where(__normalized_map == 1, -1, __normalized_map)
            __normalized_map = np.where(__normalized_map == 0, 1, __normalized_map)
            __normalized_map = __normalized_map.flatten()
            __X, __Y = np.meshgrid(map_line[0], map_line[1])
            __ones_coords = np.column_stack((__X.flatten(), __Y.flatten()))
            __kd_tree = cKDTree(__ones_coords)
            __neighbor_indices_RR = __kd_tree.query_ball_point(point, r=RR, p=2)
            __distances_RR = np.array([np.linalg.norm(__ones_coords[idx] - point) for idx in __neighbor_indices_RR])
            __sorted_indices_RR = np.argsort(__distances_RR)
            __neighbor_indices_RR = np.array(__neighbor_indices_RR)[__sorted_indices_RR]
            __max_ones = -np.inf  # Инициализация max_ones с минимально возможным значением
            corr_point = None
            for __index in __neighbor_indices_RR:
                __neighbor_coord = __ones_coords[__index]
                __dist = distance.euclidean(point, (__neighbor_coord[0], __neighbor_coord[1]))
                if __dist <= RR:
                    __neighbor_indices_R = __kd_tree.query_ball_point(__neighbor_coord, r=R, p=2)
                    __zero_indices = __kd_tree.query_ball_point(__neighbor_coord, r=R_check, p=2)
                    if np.any(__normalized_map[__zero_indices] == 0):
                        __count = 0
                    else:
                        __weights = np.array(
                            [1 - (distance.euclidean((__neighbor_coord[0], __neighbor_coord[1]),
                                                     __ones_coords[__i]) / R)
                             for __i in __neighbor_indices_R])
                        __weights /= np.sum(__weights)
                        __count = sum(__normalized_map[__neighbor_indices_R] * __weights)
                    if __count > __max_ones:
                        __max_ones = __count
                        corr_point = tuple(__neighbor_coord)
            if corr_point is None:
                corr_point = point
            return corr_point

        @staticmethod
        def points_map_line_corr(map: np.ndarray, map_line: tuple, point_line: tuple, points: tuple, RR: float,
                                 R: float, R_check: float, width: float) -> tuple:
            __normalized_map = map / 255
            __normalized_map = np.where(__normalized_map == 1, -1, __normalized_map)
            __normalized_map = np.where(__normalized_map == 0, 1, __normalized_map)
            __normalized_map = __normalized_map.flatten()
            __X, __Y = np.meshgrid(map_line[0], map_line[1])
            __ones_coords = np.column_stack((__X.flatten(), __Y.flatten()))
            __kd_tree = cKDTree(__ones_coords)

            point_1, point_2 = point_line
            line = LineString([point_1, point_2])
            direction = np.array(point_2) - np.array(point_1)
            direction = direction / np.linalg.norm(direction)
            perpendicular = np.array([-direction[1], direction[0]])
            corr_points = np.empty((0, 2), dtype=float)
            for point in points:
                rectangle_points = [
                    (point + perpendicular * RR / 2),
                    (point - perpendicular * RR / 2),
                    (point - perpendicular * RR / 2 - direction * width),
                    (point + perpendicular * RR / 2 - direction * width),
                    (point + perpendicular * RR / 2)
                ]
                rectangle = Polygon(rectangle_points)
                inside_points = np.array([p for p in __ones_coords if rectangle.contains(Point(p))]).reshape(-1, 2)
                inside_indices = np.array([np.where(np.all(__ones_coords == p, axis=1))[0][0] for p in inside_points])
                __distances_RR = np.array([np.linalg.norm(i - point) for i in inside_points])
                __sorted_indices_RR = np.argsort(__distances_RR)
                __neighbor_indices_RR = np.array(inside_indices)[__sorted_indices_RR]
                __max_ones = -np.inf
                corr_point = None
                for __index in __neighbor_indices_RR:
                    __neighbor_coord = __ones_coords[__index]
                    __dist = distance.euclidean(point, (__neighbor_coord[0], __neighbor_coord[1]))
                    if __dist <= RR:
                        __neighbor_indices_R = __kd_tree.query_ball_point(__neighbor_coord, r=R, p=2)
                        __zero_indices = __kd_tree.query_ball_point(__neighbor_coord, r=R_check, p=2)
                        if np.any(__normalized_map[__zero_indices] == 0):
                            __count = 0
                        else:
                            __weights = np.array([1 - (distance.euclidean((__neighbor_coord[0], __neighbor_coord[1]),
                                                                          __ones_coords[__i]) / R) for __i in
                                                  __neighbor_indices_R])
                            __weights /= np.sum(__weights)
                            __count = sum(__normalized_map[__neighbor_indices_R] * __weights)
                        if __count > __max_ones:
                            __max_ones = __count
                            corr_point = tuple(__neighbor_coord)
                if corr_point is None:
                    corr_point = point
                corr_points = np.append(corr_points, [corr_point], axis=0)
            return corr_points.reshape(-1, 2)

    def __create_map(self):
        if self.map_directory is not None:
            files = os.listdir(self.map_directory)
            valid_extensions = ('.png', '.PNG', '.jpeg', '.JPEG')
            valid_files = [f for f in files if f.lower().endswith(valid_extensions)]
            map_path = os.path.join(self.map_directory, valid_files[0])
            __image = cv2.imread(map_path)
            binary_map = np.zeros((__image.shape[0], __image.shape[1]), dtype=np.uint8)
            for lower_bounds, upper_bounds in zip(self.rgb_lower, self.rgb_upper):
                mask = cv2.inRange(__image, np.array(lower_bounds), np.array(upper_bounds))
                binary_map = cv2.bitwise_or(binary_map, mask)
            _, smoothed_binary_map = cv2.threshold(binary_map, 127, 255, cv2.THRESH_BINARY)
            smoothed_binary_map = np.flipud(smoothed_binary_map)
            self.map_x_line = np.linspace(self.x_min, self.x_max, num=smoothed_binary_map.shape[1])
            self.map_y_line = np.linspace(self.y_min, self.y_max, num=smoothed_binary_map.shape[0])
            self.binary_map = smoothed_binary_map

    def __create_w_pad(self):
        __n = np.floor(self.get_area / self.get_wp_area)
        if __n >= self.wp_max:
            self.logger.log_warning(f'High preliminary number of CP calculated by areas, auto-correction for limitation {self.wp_max}, was {__n}')
            __n = self.wp_max
        else:
            self.logger.log_info(f"Preliminary number of CP = {int(__n)}")
        __m = int(__n)
        _n_pad_points = []          # Нормированные координаты центров кластеров
        _a_pad_points = []          # Абсолютные координаты центров кластеров
        _a_wp_wz_points = []        # Нормированные координаты сгрупированных точек
        _n_wp_wz_points = []        # Абсолютные координаты сгрупированных точек
        _wp_wz_values = []          # Веса центров кластеров
        _pad_values = []            # Веса центров сгрупированных точек
        _wp_wz_names = []           # Имена зон
        _pad_names = []             # Имена кустов
        __points = np.copy(self.get_n_wz_points[:, :2])
        __values = np.copy(self.get_wz_values)
        __binding_names = np.copy(self.get_wz_names)
        __ires = 0
        __wp = 1
        for __iteration in range(self.wp_max_iter, 0, -1):
            __status = False
            __labels, __pad_labels, __n_pad_points, __pad_values = WZone.Cluster.create_kmeans(__points, __values, __m)
            __a_pad_points = Transform.Change.denormalize_2d(
                __n_pad_points,
                limit_min=(self.x_grid_min, self.y_grid_min),
                limit_max=(self.x_grid_max, self.y_grid_max)
            )
            if self.map_directory is not None:
                __a_pad_points = WPad.MapPoint.points_map_corr(
                    self.binary_map,
                    (self.map_x_line, self.map_y_line),
                    __a_pad_points,
                    R=self.wp_r,
                    RR=self.wp_rr,
                    R_check=self.wp_r_chek
                )
            for __i, (__label, __n_point, __a_point, __value) in enumerate(zip(__pad_labels, __n_pad_points, __a_pad_points, __pad_values)):
                if len(_n_pad_points) == 0:
                    __status = True
                    _pad_names.append(WZone.Name.create_sample_name(self.name, 'WP', __wp))
                    __a_zone_points = Transform.Change.denormalize_2d(
                        __points,
                        limit_min=(self.x_grid_min, self.y_grid_min),
                        limit_max=(self.x_grid_max, self.y_grid_max)
                    )
                    distances_1 = distance_matrix(__a_zone_points, [__a_point])
                    _n_pad_points.append(__n_point)
                    _a_pad_points.append(__a_point)
                    __mask_T = (__labels == __label) & (distances_1.flatten() <= self.wp_sector)
                    __mask_F = ~__mask_T
                    _a_wp_wz_points.append(__a_zone_points[__mask_T])
                    _n_wp_wz_points.append(__points[__mask_T])
                    _wp_wz_names.append(__binding_names[__mask_T])
                    _wp_wz_values.append(__values[__mask_T])
                    _pad_values.append(__value)
                    __points = __points[__mask_F]
                    __values = __values[__mask_F]
                    __labels = __labels[__mask_F]
                    __binding_names = __binding_names[__mask_F]
                    __m -= 1
                    __wp += 1
                else:
                    distances_0 = distance_matrix(_a_pad_points, [__a_point])
                    if np.all(distances_0 >= self.wp_distantce):
                        __status = True
                        __a_zone_points = Transform.Change.denormalize_2d(
                            __points,
                            limit_min=(self.x_grid_min, self.y_grid_min),
                            limit_max=(self.x_grid_max, self.y_grid_max)
                        )
                        distances_1 = distance_matrix(__a_zone_points, [__a_point])
                        __mask_T = (__labels == __label) & (distances_1.flatten() <= self.wp_sector)
                        __mask_F = ~__mask_T

                        if not np.any(__mask_T):
                            __status = False
                        else:
                            _n_pad_points.append(__n_point)
                            _a_pad_points.append(__a_point)
                            _pad_names.append(WZone.Name.create_sample_name(self.name, 'WP', __wp))
                            _a_wp_wz_points.append(__a_zone_points[__mask_T])
                            _n_wp_wz_points.append(__points[__mask_T])
                            _wp_wz_names.append(__binding_names[__mask_T])
                            _wp_wz_values.append(__values[__mask_T])
                            _pad_values.append(__value)
                            __points = __points[__mask_F]
                            __values = __values[__mask_F]
                            __labels = __labels[__mask_F]
                            __binding_names = __binding_names[__mask_F]
                            __m -= 1
                            __wp += 1
            if len(_n_pad_points) == __n:
                self.logger.log_info(f"Found maximum number of CP {len(_n_pad_points)} in {self.wp_max_iter- __iteration + 1} iterations")
                break
            if not __status:
                __ires += 1
                if __ires >= self.wp_const_iter:
                    self.logger.log_info(f"Stopping at {self.wp_max_iter - __iteration + 1} iterations according to the limit constraint")
                    break
            else:
                __ires = 0
            if __iteration == 0:
                self.logger.log_info(f"Stopping at {self.wp_max_iter - __iteration + 1} iterations according to max limit")
                break
            self.logger.log_info(f"Found {len(_n_pad_points)} CPs in {self.wp_max_iter - __iteration + 1} iterations")
        self._n_wp_points = np.array(_n_pad_points).reshape(-1, 2)
        self._a_wp_points = np.array(_a_pad_points).reshape(-1, 2)
        self._a_wp_wz_points = _a_wp_wz_points
        self._n_wp_wz_points = _n_wp_wz_points
        self._wp_values = np.array(_pad_values).flatten()
        self._wp_wz_values = _wp_wz_values
        self._wp_names = np.array(_pad_names)
        self._wp_wz_names = _wp_wz_names
        self._dict_n_wp_points = {name: point for name, point in zip(self.get_wp_names, self.get_n_wp_points)}
        self._dict_a_wp_points = {name: point for name, point in zip(self.get_wp_names, self.get_a_wp_points)}
        self._dict_n_wp_wz_points = {name: points for name, points in zip(self.get_wp_names, self._n_wp_wz_points)}
        self._dict_a_wp_wz_points = {name: points for name, points in zip(self.get_wp_names, self._a_wp_wz_points)}
        self._dict_wp_values = {name: value for name, value in zip(self.get_wp_names, self.get_wp_values)}
        self._dict_wp_wz_values = {name: values for name, values in zip(self.get_wp_names, self.get_wp_wz_values)}
        self._dict_wp_wz_names = {name: names for name, names in zip(self.get_wp_names, self.get_wp_wz_names)}

    def __wp_visual(self):
        if self.map_directory:
            if self.result_save:
                normalized_map = self.binary_map / 255
                normalized_map = np.where(normalized_map == 1, -1, normalized_map)
                normalized_map = np.where(normalized_map == 0, 1, normalized_map)
                aspect_ratio = (self.x_max - self.x_min) / (self.y_max - self.y_min)
                width = 800
                height = int(width / aspect_ratio)
                fig = go.Figure()
                fig = go.Figure(layout=go.Layout(
                    width=width,
                    height=height
                ))
                fig.add_trace(go.Heatmap(
                    z=normalized_map,
                    x=self.map_x_line,
                    y=self.map_y_line,
                    hoverongaps=False,
                    colorscale=[[0, 'rgb(68, 122, 173)'], [1, 'rgb(200, 200, 200)']]
                ))
                fig.write_html(f"{self.dict_path_folders['current']}\\binary_map.html")
                self.logger.log_info(f"Saving 2D binary surface map {self.dict_path_folders['current']}\\binary_map.html")
        if self.result_save:
            fig = go.Figure()
            colors = plt.cm.get_cmap('tab20', len(self._wp_names))
            for i, name in enumerate(self._wp_names):
                center_color = colors(i)[:3]
                center_color_rgb = f"rgb{tuple(int(center_color[j] * 255) for j in range(3))}"
                fig.add_trace(go.Scatter(
                    x=[self._dict_a_wp_points[name][0]],
                    y=[self._dict_a_wp_points[name][1]],
                    name=f"{name}",
                    mode='markers',
                    marker=dict(symbol='square', color=center_color_rgb, size=10)
                ))
                for points, names in zip(self._dict_a_wp_wz_points[name], self._dict_wp_wz_names[name]):
                    num_points = len(points)
                    hues = np.linspace(0, 1, num_points + 1)[:-1]  # Равномерное распределение оттенков
                    point_colors = [colorsys.hls_to_rgb(hue, center_color[1], center_color[2]) for hue in hues]
                    point_colors_rgb = [f"rgb{tuple(int(c[j] * 255) for j in range(3))}" for c in point_colors]
                    fig.add_trace(go.Scatter(
                        x=[points[0]],
                        y=[points[1]],
                        name=f"{name}-{names}",
                        mode = 'markers',
                        marker = dict(color=point_colors_rgb, size=5)
                    ))

            fig.write_html(f"{self.dict_path_folders['current']}\\binding_wp_wz.html")
            self.logger.log_info(f"Saving 2D reference map {self.dict_path_folders['current']}\\binding_wp_wz.html")

    def __call__(self, *args, **kwargs):
        WZone.__call__(self)
        self.__create_map()
        self.__create_w_pad()
        self.__wp_visual()
        
    """  Свойства кластеров """

    # region

    @property
    def get_wp_names(self) -> np.ndarray:
        """
        :options:
        Возвращает спиоск имен кустов
        :return: np.ndarray
        """
        return self._wp_names

    @property
    def get_n_wp_points(self) -> np.ndarray:
        """
        :options:
        Возвращает спиоск нормализованных координат кустов
        :return: np.ndarray
        """
        return self._n_wp_points

    @property
    def get_a_wp_points(self) -> np.ndarray:
        """
        :options:
        Возвращает спиоск абсолютных координат кустов
        :return: np.ndarray
        """
        return self._a_wp_points

    @property
    def get_wp_values(self) -> np.ndarray:
        """
        :options:
        Веса центров зон генераций траектории скважин
        :return: np.ndarray
        """
        return self._wp_values

    @property
    def get_dict_n_wp_points(self) -> Dict[str, np.ndarray]:
        """
        :options:
        Возвращает словарь имен кустов и их нормированных координат
        :return: Dict[str, np.ndarray]
        """
        return self._dict_n_wp_points

    @property
    def get_dict_a_wp_points(self) -> Dict[str, np.ndarray]:
        """
        :options:
        Возвращает словарь имен кустов и их абсолютных координат
        :return: Dict[str, np.ndarray]
        """
        return self._dict_a_wp_points

    @property
    def get_dict_wp_values(self) -> Dict[str, np.ndarray]:
        """
        :options:
        Возвращает словарь имен кустов и их веса
        :return: Dict[str, np.ndarray]
        """
        return self._dict_wp_values

    # endregion

    """ Свойства кластерных точек """

    # region

    @property
    def get_n_wp_wz_points(self) -> List[np.ndarray]:
        """
        :options:
        Двумерный массив нормированных координат центров зон, группированных по кустам
        :return: List[np.ndarray]
        """
        return self._n_wp_wz_points

    @property
    def get_a_wp_wz_points(self) -> List[np.ndarray]:
        """
        :options:
        Двумерный массив абсолютных координат центров зон, группированных по кустам
        :return: List[np.ndarray]
        """
        return self._a_wp_wz_points

    @property
    def get_wp_wz_values(self) -> List[np.ndarray]:
        """
        :options:
        Двумерный массив весов центров зон, группированных по кустам
        :return: List[np.ndarray]
        """
        return self._wp_wz_values

    @property
    def get_wp_wz_names(self) -> List[np.ndarray]:
        """
        :options:
        Двумерный массив имен зон, группированных по кустам
        :return: List[np.ndarray]
        """
        return self._wp_wz_names

    @property
    def get_dict_n_wp_wz_points(self) -> Dict[str, List[np.ndarray]]:
        """
        :options:
        Возвращает словарь имен кустов и соответсвующие нормированные координаты зон
        :return: Dict[str, List[np.ndarray]]
        """
        return self._dict_n_wp_wz_points

    @property
    def get_dict_a_wp_wz_points(self) -> Dict[str, List[np.ndarray]]:
        """
        :options:
        Возвращает словарь имен кустов и соответсвующие абсолютные координаты зон
        :return: Dict[str, List[np.ndarray]]
        """
        return self._dict_a_wp_wz_points

    @property
    def get_dict_wp_wz_values(self) -> Dict[str, List[float]]:
        """
        :options:
        Возвращает словарь имен кустов и соответсвующие веса зон
        :return: Dict[str, List[float]]
        """
        return self._dict_wp_wz_values

    @property
    def get_dict_wp_wz_names(self) -> Dict[str, List[str]]:
        """
        :options:
        Возвращает словарь имен кустов и соответсвующие имена зон
        :return: Dict[str, List[str]]
        """
        return self._dict_wp_wz_names

    @property
    def get_wp_area(self) -> float:
        return np.pi * (self.wp_sector ** 2) / 4




