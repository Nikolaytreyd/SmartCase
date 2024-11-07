from modules.transform import Transform
from config.importation import *

class WZone(Transform):
    def __init__(self, variable_file):
        Transform.__init__(self, variable_file)
        self.__area = None
        self.__n_area = None
        self._n_wz_points = None
        self._a_wz_points = None
        self._n_belong_points = None
        self._a_belong_points = None
        self._wz_values = None
        self._belong_values = None
        self._wz_names = None
        self._dict_n_wz_points = None
        self._dict_a_wz_points = None
        self._dict_n_belong_points = None
        self._dict_a_belong_points = None
        self._dict_wz_values = None
        self._dict_belong_values = None

    class Name:

        @staticmethod
        def create_sample_name(info, name: str, order: int) -> str:
            return f"{info}_{name}{WZone.Name.chek_name(order)}"

        @staticmethod
        def chek_name(order: int) -> str:
            """
            :param order: Порядок;
            :return: Четырехзначный запись порядка
            """
            if order < 10:
                return f'00{order}'
            elif order < 100:
                return f'0{order}'
            else:
                return f'{order}'

    class Cluster:
        @staticmethod
        def interpolate_weight(coordinates: np.ndarray, weights: np.ndarray, central_point: np.ndarray):
            return griddata(coordinates, weights, [central_point], method='nearest')[0]

        @staticmethod
        def create_kmeans(points: np.ndarray, values: np.ndarray, n: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            """
            Options: Кластеризация точек по kmeans, сортировка по убыванию привлекательности кластеров
            Args:
                points: Массив координат точек
                values: Массив весов каждой точки
                n: Количество кластеров
            Returns: метки кластера, координаты центров кластеров, веса кластеров
            """
            __kmeans = KMeans(n_clusters=n, n_init=1, max_iter=300)
            __kmeans.fit(points, sample_weight=values)
            __center_points = __kmeans.cluster_centers_
            labels = __kmeans.labels_
            __cluster_sizes = np.bincount(labels)
            __center_weights = np.array(
                [WZone.Cluster.interpolate_weight(points, values, center) * __cluster_sizes[i] for i, center in
                 enumerate(__center_points)])
            __center_labels = np.arange(n)
            __sorted_indices = np.argsort(__center_weights)[::-1]
            sorted_centers = __center_points[__sorted_indices]
            center_weights = __center_weights[__sorted_indices]
            center_labels = __center_labels[__sorted_indices]
            return labels, center_labels, sorted_centers, center_weights

        @staticmethod
        def create_gmm(points: np.ndarray, values: np.ndarray, n: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            """
            Options: Кластеризация точек по алгоритму гауссовской смеси, сортировка по убыванию привлекательности кластеров
            Args:
                points: Массив координат точек
                values: Массив весов каждой точки
                n: Количество кластеров

            Returns: метки кластера, координаты центров кластеров, веса кластеров
            """
            __gmm = GaussianMixture(n_components=n, n_init=50, random_state=0, max_iter=300, reg_covar=pow(10, -3)).fit(
                points)
            __center_points = __gmm.means_
            labels = __gmm.predict(points)
            __cluster_sizes = np.bincount(labels)
            __center_weights = np.array([np.sum(values[labels == i]) for i in range(__gmm.n_components)])
            __center_labels = np.arange(n)
            __sorted_indices = np.argsort(__center_weights)[::-1]
            sorted_centers = __center_points[__sorted_indices]
            center_weights = __center_weights[__sorted_indices]
            center_labels = __center_labels[__sorted_indices]
            return labels, center_labels, sorted_centers, center_weights

        @staticmethod
        def create_simple_kmeans(points: np.ndarray, values: np.ndarray or None, n: int, option=False):
            __kmeans = KMeans(n_clusters=n, n_init=1, max_iter=300)
            if not values is None:
                # __kmeans.fit(points, sample_weight=values)
                __kmeans.fit(points)
                cluster_labels = __kmeans.labels_
                cluster_points = __kmeans.cluster_centers_
                if not option:
                    __cluster_sizes = np.bincount(cluster_labels)
                    cluster_weights = np.array(
                        [WZone.Cluster.interpolate_weight(points, values, center) * __cluster_sizes[i] for i, center in
                         enumerate(cluster_points)])
                else:
                    cluster_weights = np.bincount(cluster_labels, weights=values)
                return cluster_labels, cluster_points, cluster_weights
            else:
                __kmeans.fit(points)
                cluster_points = __kmeans.cluster_centers_
                cluster_labels = __kmeans.labels_
                return cluster_labels, cluster_points

    def __definition_area(self) -> None:
        """
        :options:
        Расчет площади через алгоритм сжатия контуров - алгоритм Дугласа-Пекера
        Определение атрибутов _area и _n_area
        Returns: None
        """
        __points = MultiPoint(list(zip(self.get_dfn_coords[:, 0], self.get_dfn_coords[:, 1])))
        if __points.geom_type == 'MultiPoint':
            __points_coords = [list(point.coords) for point in __points.geoms]
            __points_coords = [item for sublist in __points_coords for item in sublist]  # Flatten the list
        else:
            __points_coords = list(__points.coords)
        hull = ConvexHull(__points_coords)
        hull_points = [__points_coords[vertex] for vertex in hull.vertices]
        __polygon = Polygon(hull_points)
        __rectangle = abs(self.x_max - self.x_min) * abs(self.y_max - self.y_min)
        self.__area = __polygon.area * __rectangle
        self.__n_area = __polygon.area
        self.logger.log_info('The area for division into zones has been calculated')

    def __create_w_zone(self) -> None:
        self.logger.log_info('Initialization of the algorithm for splitting an object into zones without regions')
        __n = np.floor(self.get_area / self.get_w_area)
        if __n >= self.wz_max:
            self.logger.log_warning(f'High preliminary number of zones calculated by areas, auto-correcting for limitation {self.wz_max}, was {__n}')
            __n = self.wz_max
        __m = int(__n)
        _n_zone_points = []      # Нормированные координаты центров кластеров
        _a_zone_points = []      # Абсолютные координаты центров кластеров
        _n_belong_points = []    # Нормированные координаты сгрупированных точек
        _a_belong_points = []    # Абсолютные координаты сгрупированных точек
        _zone_values = []        # Веса центров кластеров
        _belong_values = []      # Веса центров сгрупированных точек
        _zone_names = []         # Имена зон
        __wz = 1                 # Порядок расчтета при создании имени зоны
        __ires = 0               # Порядок безрезультатных итераций
        __points = np.copy(self.get_dfn_coords)
        __values = np.copy(self.get_dfn_values)
        for __iteration in range(self.wz_max_iter, 0, -1):
            __status = False
            __labels, __center_labels, __n_center_points, __center_values = WZone.Cluster.create_kmeans(__points, __values, __m)
            __a_center_points = Transform.Change.denormalize_3d(
                __n_center_points,
                limit_min=(self.x_grid_min, self.y_grid_min, self.z_grid_min),
                limit_max=(self.x_grid_max, self.y_grid_max, self.z_grid_max),
                z_normalize=self.z_normalize
            )
            for __i, (__label, __n_point, __a_point, __value) in enumerate(zip(__center_labels, __n_center_points,  __a_center_points, __center_values)):
                if len(_n_zone_points) == 0:
                    __status = True
                    _zone_names.append(WZone.Name.create_sample_name(self.name, 'W', __wz))
                    __a_points = Transform.Change.denormalize_3d(
                        __points,
                        limit_min=(self.x_grid_min, self.y_grid_min, self.z_grid_min),
                        limit_max=(self.x_grid_max, self.y_grid_max, self.z_grid_max),
                        z_normalize=self.z_normalize
                    )
                    distances_1 = distance_matrix(__a_points, [__a_point])
                    _n_zone_points.append(__n_point)
                    _a_zone_points.append(__a_point)
                    __mask_T = (__labels == __label) & (distances_1.flatten() <= self.wz_sector)
                    __mask_F = ~__mask_T
                    _n_belong_points.append(__points[__mask_T])
                    _a_belong_points.append(__a_points[__mask_T])
                    _zone_values.append(__value)
                    _belong_values.append(__values[__mask_T])
                    __points = __points[__mask_F]
                    __values = __values[__mask_F]
                    __labels = __labels[__mask_F]
                    __m -= 1
                    __wz += 1
                else:
                    distances_0 = distance_matrix(_a_zone_points, [__a_point])
                    if np.all(distances_0 >= self.wz_distantce):
                        __status = True
                        _zone_names.append(WZone.Name.create_sample_name(self.name, 'W', __wz))
                        __a_points = Transform.Change.denormalize_3d(
                            __points,
                            limit_min=(self.x_grid_min, self.y_grid_min, self.z_grid_min),
                            limit_max=(self.x_grid_max, self.y_grid_max, self.z_grid_max),
                            z_normalize=self.z_normalize
                        )
                        distances_1 = distance_matrix(__a_points, [__a_point])
                        _n_zone_points.append(__n_point)
                        _a_zone_points.append(__a_point)
                        __mask_T = (__labels == __label) & (distances_1.flatten() <= self.wz_sector)
                        __mask_F = ~__mask_T
                        _n_belong_points.append(__points[__mask_T])
                        _a_belong_points.append(__a_points[__mask_T])
                        _zone_values.append(__value)
                        _belong_values.append(__values[__mask_T])
                        __points = __points[__mask_F]
                        __values = __values[__mask_F]
                        __labels = __labels[__mask_F]
                        __m -= 1
                        __wz += 1
            if len(_n_zone_points) == __n:
                self.logger.log_info(f"The maximum number of zones was found in {self.wz_max_iter- __iteration + 1} iterations")
                break
            if not __status:
                __ires += 1
                if __ires >= self.wz_const_iter:
                    self.logger.log_info(f"Stop at {self.wz_max_iter - __iteration + 1} iterations according to the limit constraint")
                    break
            else:
                __ires = 0
            if __iteration == 0:
                self.logger.log_info(f"Stop at {self.wz_max_iter - __iteration + 1} iterations according to the max constraint")
                break
            self.logger.log_info(f"Found {len(_n_zone_points)} zone(s) in {self.wz_max_iter - __iteration + 1} iterations")
        self._n_wz_points = np.array(_n_zone_points).reshape(-1, 3)
        self._a_wz_points = np.array(_a_zone_points).reshape(-1, 3)
        self._n_belong_points = _n_belong_points
        self._a_belong_points = _a_belong_points
        self._wz_values = np.array(_zone_values).flatten()
        self._belong_values = _belong_values
        self._wz_names = np.array(_zone_names)
        self._dict_n_wz_points = {name: point for name, point in zip(self.get_wz_names, self.get_n_wz_points)}
        self._dict_a_wz_points = {name: point for name, point in zip(self.get_wz_names, self.get_a_wz_points)}
        self._dict_n_belong_points = {name: points for name, points in zip(self.get_wz_names, self._n_belong_points)}
        self._dict_a_belong_points = {name: points for name, points in zip(self.get_wz_names, self._a_belong_points)}
        self._dict_wz_values = {name: value for name, value in zip(self.get_wz_names, self.get_wz_values)}
        self._dict_belong_values = {name: values for name, values in zip(self.get_wz_names, self.get_belong_values)}

    def __create_w_zone_regions(self) -> None:
        self.logger.log_info('Initialization of the algorithm for dividing an object into zones with regions')
        _n_zone_points = []      # Нормированные координаты центров кластеров
        _a_zone_points = []      # Абсолютные координаты центров кластеров
        _n_belong_points = []    # Нормированные координаты сгрупированных точек
        _a_belong_points = []    # Абсолютные координаты сгрупированных точек
        _zone_values = []        # Веса центров кластеров
        _belong_values = []      # Веса центров сгрупированных точек
        _zone_names = []         # Имена зон
        __total = len(self.get_df_regions)  # Предполагается, что это свойство возвращает список регионов

        __n = np.floor(self.get_area / self.get_w_area)
        if __n >= self.wz_max:
            self.logger.log_warning(f'High preliminary number of zones calculated by areas, auto-correcting for limitation {self.wz_max}, was {round(__n)}')
            __n = self.wz_max

        __wz = 1  # Порядок расчтета при создании имени зоны
        for region in np.unique(self.get_df_regions):
            region_status = False
            __ires = 0  # Порядок безрезультатных итераций
            __region_mask = self.get_df_regions == region
            __region_count = np.sum(__region_mask)
            if __region_count == 0:
                self.logger.log_warning('Region {region} is not defined by dots, skipping ...')
                continue
            __points = np.copy(self.get_dfn_coords[__region_mask])
            __values = np.copy(self.get_dfn_values[__region_mask])
            __m = int(__n * __region_count / __total)
            if __m <= 2:
                self.logger.log_warning(f'Region {region} occupies a low {round(__region_count / __total, 2)} share of the entire cube, skip ...')
                continue
            for __iteration in range(self.wz_max_iter, 0, -1):
                __status = False
                __labels, __center_labels, __n_center_points, __center_values = WZone.Cluster.create_kmeans(__points, __values, __m)
                __a_center_points = Transform.Change.denormalize_3d(
                    __n_center_points,
                    limit_min=(self.x_grid_min, self.y_grid_min, self.z_grid_min),
                    limit_max=(self.x_grid_max, self.y_grid_max, self.z_grid_max),
                    z_normalize=self.z_normalize
                )
                for __i, (__label, __n_point, __a_point, __value) in enumerate(zip(__center_labels, __n_center_points, __a_center_points, __center_values)):
                    if not region_status:
                        region_status = True
                        __status = True
                        _zone_names.append(WZone.Name.create_sample_name(self.name, 'W', __wz))
                        __a_points = Transform.Change.denormalize_3d(
                            __points,
                            limit_min=(self.x_grid_min, self.y_grid_min, self.z_grid_min),
                            limit_max=(self.x_grid_max, self.y_grid_max, self.z_grid_max),
                            z_normalize=self.z_normalize
                        )
                        distances_1 = distance_matrix(__a_points, [__a_point])
                        _n_zone_points.append(__n_point)
                        _a_zone_points.append(__a_point)
                        __mask_T = (__labels == __label) & (distances_1.flatten() <= self.wz_sector)
                        __mask_F = ~__mask_T
                        _n_belong_points.append(__points[__mask_T])
                        _a_belong_points.append(__a_points[__mask_T])
                        _zone_values.append(__value)
                        _belong_values.append(__values[__mask_T])
                        __points = __points[__mask_F]
                        __values = __values[__mask_F]
                        __labels = __labels[__mask_F]
                        __m -= 1
                        __wz += 1
                    else:
                        distances_0 = distance_matrix(_a_zone_points, [__a_point])
                        if np.all(distances_0 >= self.wz_distantce):
                            __status = True
                            _zone_names.append(WZone.Name.create_sample_name(self.name, 'W', __wz))
                            __a_points = Transform.Change.denormalize_3d(
                                __points,
                                limit_min=(self.x_grid_min, self.y_grid_min, self.z_grid_min),
                                limit_max=(self.x_grid_max, self.y_grid_max, self.z_grid_max),
                                z_normalize=self.z_normalize
                            )
                            distances_1 = distance_matrix(__a_points, [__a_point])
                            _n_zone_points.append(__n_point)
                            _a_zone_points.append(__a_point)
                            __mask_T = (__labels == __label) & (distances_1.flatten() <= self.wz_sector)
                            __mask_F = ~__mask_T
                            _n_belong_points.append(__points[__mask_T])
                            _a_belong_points.append(__a_points[__mask_T])
                            _zone_values.append(__value)
                            _belong_values.append(__values[__mask_T])
                            __points = __points[__mask_F]
                            __values = __values[__mask_F]
                            __labels = __labels[__mask_F]
                            __m -= 1
                            __wz += 1
                if __m <= 0:
                    self.logger.log_info(f"The maximum number of zones found in {self.wz_max_iter - __iteration + 1} iterations in region {region}")
                    break
                if not __status:
                    __ires += 1
                    if __ires >= self.wz_const_iter:
                        self.logger.log_info(f"Stop at {self.wz_max_iter - __iteration + 1} iterations according to the limit constraint in region {region}")
                        break
                else:
                    __ires = 0
                if __iteration == 0:
                    self.logger.log_info(f"Stop at {self.wz_max_iter - __iteration + 1} iterations according to the maximum constraint in region {region}")
                    break
                self.logger.log_info(f"Found {len(_n_zone_points)} zone(s) in {self.wz_max_iter - __iteration + 1} iterations in region {region}")
            if len(_n_zone_points) == __n:
                self.logger.log_info(f"The maximum number of zones found in {self.wz_max_iter - __iteration + 1} iterations in region {region}")
                break
        self._n_wz_points = np.array(_n_zone_points).reshape(-1, 3)
        self._a_wz_points = np.array(_a_zone_points).reshape(-1, 3)
        self._n_belong_points = _n_belong_points
        self._a_belong_points = _a_belong_points
        self._wz_values = np.array(_zone_values).flatten()
        self._belong_values = _belong_values
        self._wz_names = np.array(_zone_names)
        self._dict_n_wz_points = {name: point for name, point in zip(self.get_wz_names, self.get_n_wz_points)}
        self._dict_a_wz_points = {name: point for name, point in zip(self.get_wz_names, self.get_a_wz_points)}
        self._dict_n_belong_points = {name: points for name, points in zip(self.get_wz_names, self._n_belong_points)}
        self._dict_a_belong_points = {name: points for name, points in zip(self.get_wz_names, self._a_belong_points)}
        self._dict_wz_values = {name: value for name, value in zip(self.get_wz_names, self.get_wz_values)}
        self._dict_belong_values = {name: values for name, values in zip(self.get_wz_names, self.get_belong_values)}

    def __wz_visual(self):
        if self.result_save:
            __fig = go.Figure()
            __colors = px.colors.qualitative.Plotly
            __color_list = __colors[:len(self.get_a_belong_points)]
            __all_a_belong_points = np.concatenate(self.get_a_belong_points)
            for __i, __points in enumerate(self.get_a_belong_points):
                __fig.add_trace(go.Scatter3d(
                    x=__points[:, 0],
                    y=__points[:, 1],
                    z=__points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=__color_list[__i % len(__color_list)],
                        opacity=0.2
                    )
                ))
            __fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False, range=[self.x_min, self.x_max]),
                    yaxis=dict(visible=False, range=[self.y_min, self.y_max]),
                    zaxis=dict(visible=False, range=[self.z_max + 200, self.z_min - 200]),
                    bgcolor='rgba(255,255,255,1)',
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            __fig.write_html(f"{self.dict_path_folders['current']}\wzone.html")
            self.logger.log_info(f"Saving 3D Zone Cube {self.dict_path_folders['current']}\wzone.html")

            __fig = go.Figure()
            __fig.add_trace(go.Scatter3d(
                x=self.get_dfn_coords[:, 0],
                y=self.get_dfn_coords[:, 1],
                z=self.get_dfn_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.get_dfn_values,
                    opacity=0.2
                ),
                hovertemplate='<b>X</b>: %{x}<br><b>Y</b>: %{y}<br><b>Z</b>: %{z}<br><b>Value</b>: %{marker.color}<extra></extra>'
            ))
            __fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False, range=[0, 1]),
                    yaxis=dict(visible=False, range=[0, 1]),
                    zaxis=dict(visible=False, range=[1, 0]),
                    bgcolor='rgba(255,255,255,1)',
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            __fig.write_html(f"{self.dict_path_folders['current']}\density_property.html")
            self.logger.log_info(f"Saving a 3D density cube {self.dict_path_folders['current']}\density property.html")
            if self.regions_directory is not None:
                __fig = go.Figure()
                __fig.add_trace(go.Scatter3d(
                    x=self.get_dfn_coords[:, 0],
                    y=self.get_dfn_coords[:, 1],
                    z=self.get_dfn_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=self.get_df_regions,
                        opacity=0.2
                    ),
                    hovertemplate='<b>X</b>: %{x}<br><b>Y</b>: %{y}<br><b>Z</b>: %{z}<br><b>Region</b>: %{marker.color}<extra></extra>'
                ))
                __fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False, range=[0, 1]),
                        yaxis=dict(visible=False, range=[0, 1]),
                        zaxis=dict(visible=False, range=[1, 0]),
                        bgcolor='rgba(255,255,255,1)',
                        aspectratio=dict(x=1, y=1, z=1)
                    ),
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                __fig.write_html(f"{self.dict_path_folders['current']}\\regions.html")
                self.logger.log_info(f"Saving 3D cube regions {self.dict_path_folders['current']}\\regions.html")

    def __call__(self, *args, **kwargs):
        Transform.__call__(self)
        self.__definition_area()
        if self.regions_directory is not None:
            self.__create_w_zone_regions()
        else:
            self.__create_w_zone()
        self.__wz_visual()

    @property
    def get_area(self) -> float:
        """
        :options:
        Абсолютная площадь распраостранения точек в кубе свойства
        :return: float
        """
        return self.__area

    @property
    def get_w_area(self) -> float:
        return np.pi * (self.wz_sector ** 2) / 4

    """  Свойства кластеров """

    # region
    @property
    def get_wz_names(self) -> np.ndarray:
        """
        :options:
        Возвращает спиоск имен зон
        :return: np.ndarray
        """
        return self._wz_names

    @property
    def get_n_wz_points(self) -> np.ndarray:
        """
        :options:
        Нормированные координаты центров зон генераций траектории скважин

        :return: np.ndarray
        """
        return self._n_wz_points

    @property
    def get_a_wz_points(self) -> np.ndarray:
        """
        :options:
        Абсолютные координаты центров зон генераций траектории скважин

        :return: np.ndarray
        """
        return self._a_wz_points

    @property
    def get_wz_values(self) -> np.ndarray:
        """
        :options:
        Веса центров зон генераций траектории скважин
        :return: np.ndarray
        """
        return self._wz_values

    @property
    def get_dict_n_wz_points(self) -> Dict[str, np.ndarray]:
        """
        :options:
        Словарь из имен зон и их нормализованных координат центр зон
        :return: Dict[str, np.ndarray]
        """
        return self._dict_n_wz_points

    @property
    def get_dict_a_wz_points(self) -> Dict[str, np.ndarray]:
        """
        :options:
        Словарь из имен зон и их абсолютных координат центр зон
        :return: Dict[str, np.ndarray]
        """
        return self._dict_a_wz_points

    @property
    def get_dict_wz_values(self) -> Dict[str, np.ndarray]:
        """
        :options:
        Словарь из имен зон и их весов центр зон
        :return: Dict[str, np.ndarray]
        """
        return self._dict_wz_values

    # endregion

    """  Свойства кластерных точек """

    # region

    @property
    def get_n_belong_points(self) -> List[np.ndarray]:
        """
        :options:
        Нормированные координаты точек задающих траекторию скважины
        :return: List[np.ndarray]
        """
        return self._n_belong_points

    @property
    def get_a_belong_points(self) -> List[np.ndarray]:
        """
        :options:
        Абсолютные координаты точек задающих траекторию скважины
        :return: List[np.ndarray]:
        """
        return self._a_belong_points

    @property
    def get_belong_values(self) -> List[np.ndarray]:
        """
        :options:
        Веса точек задающих траекторию скважины
        :return: List[np.ndarray]:
        """
        return self._belong_values

    @property
    def get_dict_n_belong_points(self) -> Dict[str, List[np.ndarray]]:
        """
        :options:
        Словарь из имен зон и их нормализованных координат точек зон
        :return: Dict[str, list]
        """
        return self._dict_n_belong_points

    @property
    def get_dict_a_belong_points(self) -> Dict[str, List[np.ndarray]]:
        """
        :options:
        Словарь из имен зон и их абсолютных координат точек зон
        :return: Dict[str, List[np.ndarray]]
        """
        return self._dict_a_belong_points

    @property
    def get_dict_belong_values(self) -> Dict[str, List[float]]:
        """
        :options:
        Словарь из имен зон и их весов точек зон
        :return: Dict[str, List[float]]
        """
        return self._dict_belong_values

    # endregion