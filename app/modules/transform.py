import numpy as np

from modules.grid import Grid
from config.importation import *


class Transform(Grid):

    def __init__(self, variable_file):
        Grid.__init__(self, variable_file)
        self.__n_coords, self.__n_values = None, None
        self.__f_coords = None
        self.__fn_coords, self.__fn_values, self.__f_regions = None, None, None
        self.__dfn_coords, self.__dfn_values, self.__df_regions = None, None, None
        self.x_grid_min = None
        self.x_grid_max = None
        self.y_grid_min = None
        self.y_grid_max = None
        self.z_grid_min = None
        self.z_grid_max = None

    class Change:
        @staticmethod
        def normalizes_2d(points: list, limit_min: tuple, limit_max: tuple) -> list:
            __x_min, __y_min = limit_min
            __x_max, __y_max = limit_max
            normal_points = []
            for __points in points:
                denormal_local_points = np.empty(__points)
                __point_x = __points[:, 0]
                __point_y = __points[:, 1]
                x_normalized = (__point_x - np.min(__point_x)) / (np.max(__point_x) - np.min(__point_x))
                y_normalized = (__point_y - np.min(__point_y)) / (np.max(__point_y) - np.min(__point_y))
                x_denormalized = x_normalized * (__x_max - __x_min) + __x_min
                y_denormalized = y_normalized * (__y_max - __y_min) + __y_min

                normal_points.append(np.array([x_denormalized, y_denormalized]))

            return normal_points

        @staticmethod
        def normalizes_3d(points: list, limit_min: tuple, limit_max: tuple) -> list:
            __x_min, __y_min, __z_min = limit_min
            __x_max, __y_max, __z_max = limit_max
            normal_points = []
            for __points in points:
                __point_x = __points[:, 0]
                __point_y = __points[:, 1]
                __point_z = __points[:, 2]
                x_normalized = (__point_x - np.min(__point_x)) / (np.max(__point_x) - np.min(__point_x))
                y_normalized = (__point_y - np.min(__point_y)) / (np.max(__point_y) - np.min(__point_y))
                z_normalized = (__point_z - np.min(__point_z)) / (np.max(__point_z) - np.min(__point_z))
                x_denormalized = x_normalized * (__x_max - __x_min) + __x_min
                y_denormalized = y_normalized * (__y_max - __y_min) + __y_min
                z_denormalized = z_normalized * (__z_max - __z_min) + __z_min
                normal_points.append(np.array([x_denormalized, y_denormalized, z_denormalized]))
            return normal_points

        @staticmethod
        def normalize_2d(point: np.ndarray, limit_min: tuple, limit_max: tuple) -> np.ndarray:
            __x_min, __y_min = limit_min
            __x_max, __y_max = limit_max
            __point_x = point[:, 0]
            __point_y = point[:, 1]
            x_normalized = (__point_x - np.min(__point_x)) / (np.max(__point_x) - np.min(__point_x))
            y_normalized = (__point_y - np.min(__point_y)) / (np.max(__point_y) - np.min(__point_y))
            x_denormalized = x_normalized * (__x_max - __x_min) + __x_min
            y_denormalized = y_normalized * (__y_max - __y_min) + __y_min
            return np.column_stack((x_denormalized, y_denormalized))

        @staticmethod
        def normalize_3d(point: np.ndarray, limit_min: tuple, limit_max: tuple) -> np.ndarray:
            __x_min, __y_min, __z_min = limit_min
            __x_max, __y_max, __z_max = limit_max
            __point_x = point[:, 0]
            __point_y = point[:, 1]
            __point_z = point[:, 2]
            x_normalized = (__point_x - np.min(__point_x)) / (np.max(__point_x) - np.min(__point_x))
            y_normalized = (__point_y - np.min(__point_y)) / (np.max(__point_y) - np.min(__point_y))
            z_normalized = (__point_z - np.min(__point_z)) / (np.max(__point_z) - np.min(__point_z))
            x_denormalized = x_normalized * (__x_max - __x_min) + __x_min
            y_denormalized = y_normalized * (__y_max - __y_min) + __y_min
            z_denormalized = z_normalized * (__z_max - __z_min) + __z_min
            return np.column_stack((x_denormalized, y_denormalized, z_denormalized))

        @staticmethod
        def normalize_1d(values: np.ndarray) -> np.ndarray:
            return (values - np.min(values)) / (np.max(values) - np.min(values))

        @staticmethod
        def denormalizes_2d(points: np.ndarray, limit_min: tuple, limit_max: tuple) -> list:
            __x_min, __y_min = limit_min
            __x_max, __y_max = limit_max
            denormal_points = []
            for __points in points:
                denormal_local_points = np.empty(__points)
                __point_x = __points[:, 0]
                __point_y = __points[:, 1]
                __d_point_x = __point_x * (__x_max - __x_min) + __x_min
                __d_point_y = __point_y * (__y_max - __y_min) + __y_min
                denormal_points.append(np.array([__d_point_x, __d_point_y]))
            return denormal_points

        @staticmethod
        def denormalizes_3d(points: list, limit_min: tuple, limit_max: tuple) -> list:
            __x_min, __y_min, __z_min = limit_min
            __x_max, __y_max, __z_max = limit_max
            denormal_points = []
            for __i, __points in enumerate(points):
                denormal_local_points = np.empty(__points)
                __point_x = __points[:, 0]
                __point_y = __points[:, 1]
                __point_z = __points[:, 2]
                __d_point_x = __point_x * (__x_max - __x_min) + __x_min
                __d_point_y = __point_y * (__y_max - __y_min) + __y_min
                __d_point_z = __point_z * (__z_max - __z_min) + __z_min
                denormal_points.append(np.array([__d_point_x, __d_point_y, __d_point_z]))
            return denormal_points

        @staticmethod
        def denormalize_2d(point: np.ndarray, limit_min: tuple, limit_max: tuple) -> np.ndarray:
            __x_min, __y_min = limit_min
            __x_max, __y_max = limit_max
            __point_x = point[:, 0]
            __point_y = point[:, 1]
            __d_point_x = __point_x * (__x_max - __x_min) + __x_min
            __d_point_y = __point_y * (__y_max - __y_min) + __y_min
            return np.column_stack((__d_point_x, __d_point_y))

        @staticmethod
        def denormalize_3d(point: np.ndarray, limit_min: tuple, limit_max: tuple, z_normalize: float) -> np.ndarray:
            __x_min, __y_min, __z_min = limit_min
            __x_max, __y_max, __z_max = limit_max
            __point_x = point[:, 0]
            __point_y = point[:, 1]
            __point_z = point[:, 2]
            __d_point_x = __point_x * (__x_max - __x_min) + __x_min
            __d_point_y = __point_y * (__y_max - __y_min) + __y_min
            __d_point_z = __point_z * (__z_max - __z_min) / z_normalize + __z_min
            return np.column_stack((__d_point_x, __d_point_y, __d_point_z))

    # endregion

    def __normalize(self) -> None:
        """
        :options:
        Определение атрибутов n_coords \n
        Приведение к равнозначному диапазону вариации по 3 трем направлениям
        Returns: None
        """
        limit_min = (0, 0, 0)
        limit_max = (1, 1, self.z_normalize)
        self.__n_coords = Transform.Change.normalize_3d(
            self.get_coords,
            limit_min,
            limit_max
        )
        self.logger.log_info(f'Normalize the grid along the x axis - [{limit_min[0]}; {limit_max[0]}]')
        self.logger.log_info(f'Normalize the grid along the y axis - [{limit_min[1]}; {limit_max[1]}]')
        if self.z_normalize >= 0.8:
            self.logger.log_warning('High range of variation of normalized Z: '
                                    'if the model is represented by one layer, there are risks of dividing the layer into zones vertically')
        if self.z_normalize <= 0.2:
            self.logger.log_warning('Low range of normalized variation Z: '
                                    'if the model is represented by several layers, there are risks of defining layers into one layer')
        self.logger.log_info(f'Normalize the grid along the z - [{limit_min[2]}; {limit_max[2]}]')
        self.__n_values = Transform.Change.normalize_1d(self.get_values)
        self.logger.log_info(f'Normalization of property cube - [0; 1]')

    def __selection(self) -> None:
        """
        :options:
        Определение атрибутов f__values и f__coords \n
        fn_values: Трехмерный массив отфильтрованных точек с столбцами X, Y, Z  \n
        fn_coords: Трехмерный массив значений отфильтрованных точек \n
        Returns: None
        """
        __non_zero_mask = self.__n_values != 0
        __non_zero_values = self.__n_values[__non_zero_mask]
        __non_zero_coords = self.__n_coords[__non_zero_mask]
        __non_zero_non_normal_coords = self.get_coords[__non_zero_mask]
        __filtered_values = median_filter(__non_zero_values, size=2)
        __threshold = np.percentile(__filtered_values.flatten(), self.selection)
        __indices = np.argwhere(__filtered_values >= __threshold)
        self.__fn_values = __filtered_values[__filtered_values >= __threshold].flatten()
        self.__fn_coords = __non_zero_coords[__indices[:, 0], :]
        self.__f_coords = __non_zero_non_normal_coords[__indices[:, 0], :]
        self.x_grid_min = self.x_min
        self.x_grid_max = self.x_max
        self.y_grid_min = self.y_min
        self.y_grid_max = self.y_max
        self.z_grid_min = self.z_min
        self.z_grid_max = self.z_max
        if self.regions_directory is not None:
            __non_zero_regions = self.get_regions[__non_zero_mask]
            self.__f_regions = __non_zero_regions[__filtered_values >= __threshold].flatten()
        else:
            self.__f_regions = np.zeros(self.__fn_values.shape)

    def __density(self) -> None:
        """
        :options:
        Определение атрибутов nf__coords \n
        Приведение плотностному распределнию точек grid с учетом их значений
        Returns: None

        """
        self.logger.log_info(f"The process of point density distribution has begun ...")
        __tree = cKDTree(self.__fn_coords)
        __dfn_coords = np.empty((self.grid_points, 3))
        __dfn_values = np.empty(self.grid_points)
        __df_regions = np.empty(self.grid_points)
        __status = 0
        __normalized_probabilities = self.__fn_values / np.sum(self.__fn_values)
        abs_ceil_x = (self.x_max - self.x_min) / self.x_n
        abs_ceil_y = (self.y_max - self.y_min) / self.y_n
        abs_ceil_z = (self.z_max - self.z_min) / self.z_n
        __ceil_x = abs_ceil_x / (self.x_max - self.x_min)
        __ceil_y = abs_ceil_y / (self.y_max - self.y_min)
        __ceil_z = abs_ceil_z / (self.z_max - self.z_min) * self.z_normalize  # Учитываем диапазон 0-0.1 для z
        for i in range(self.grid_points):
            if (i + 1) % (self.grid_points // 10) == 0:
                __status += 10
                self.logger.log_info(f'Creating a population of points on {round(__status)} %')
                if __status > 100:
                    __status = 100
            __idx = np.random.choice(np.arange(self.__fn_values.size), p=__normalized_probabilities)
            __point = self.__fn_coords[__idx]
            __x = np.random.uniform(__point[0] - __ceil_x/2, __point[0] + __ceil_x/2)
            __y = np.random.uniform(__point[1] - __ceil_y/2, __point[1] + __ceil_y/2)
            __z = np.random.uniform(__point[2] - __ceil_z/2, __point[2] + __ceil_z/2)
            __dfn_coords[i] = np.array([__x, __y, __z])
            __dfn_values[i] = self.__fn_values[__idx]
            __df_regions[i] = self.__f_regions[__idx]
        self.__dfn_values = __dfn_values
        self.__dfn_coords = __dfn_coords
        self.__df_regions = __df_regions

    def __call__(self, *args, **kwargs):
        Grid.__call__(self)
        self.__normalize()
        self.__selection()
        self.__density()

    @property
    def get_dfn_coords(self) -> np.ndarray:
        return self.__dfn_coords

    @property
    def get_dfn_values(self) -> np.ndarray:
        return self.__dfn_values

    @property
    def get_df_regions(self) -> np.ndarray:
        return self.__df_regions

    @property
    def get_ddnf_coords(self) -> np.ndarray:
        return Transform.Change.denormalize_3d(
            self.get_dfn_coords,
            limit_min=(self.x_min, self.y_min, self.z_min),
            limit_max=(self.x_max, self.y_max, self.z_max),
            z_normalize=self.z_normalize
        )



