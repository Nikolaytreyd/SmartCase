import numpy as np

from config.importation import *
from config.directiory import Direct

class Grid(Direct):
    def __init__(self, variable_file):
        Direct.__init__(self, variable_file)
        self.__file_property_names = []
        self.__file_property_path = []
        self.__file_regions_path = None
        self.__file_mask_path = None
        self.__file_coord_path = None

        self.x_min, self.x_n, self.x_max = None, None, None
        self.y_min, self.y_n, self.y_max = None, None, None
        self.z_min, self.z_n, self.z_max = None, None, None

        self.grid_cube = None
        self.map_cube = None
        self.region_cube = None
        self.coords = None

    def __read_files(self):
        # Записываем полный путь к кубам свойств
        if os.path.exists(self.property_directory):
            for filename in os.listdir(self.property_directory):
                file_path = os.path.join(self.property_directory, filename)
                if os.path.isfile(file_path):
                    self.__file_property_names.append(filename)
                    self.__file_property_path.append(file_path)
        else:
            self.logger.log_error(f'There are no loaded property cubes at path {self.property_directory}')
            raise AssertionError(f'There are no loaded property cubes at path {self.property_directory}')

        # Читаем кубы регионов
        if os.path.exists(self.regions_directory):
            for num, filename in enumerate(os.listdir(self.regions_directory)):
                if num >= 1:
                    if self.auto_tuning:
                        self.logger.log_warning(f'There are several region cubes along the path {self.regions_directory}, the file will be selected as the basis - {filename}')
                        break
                    else:
                        self.logger.log_error(f'There are several region cubes along the path {self.regions_directory}. It is necessary to leave only one region cube')
                file_path = os.path.join(self.regions_directory, filename)
                if os.path.isfile(file_path):
                    self.__file_regions_path = file_path
        else:
            self.logger.log_warning(f'There are no region property cubes downloaded at path {self.regions_directory}')

        # Читаем кубы фильтров
        if os.path.exists(self.mask_directory):
            for num, filename in enumerate(os.listdir(self.mask_directory)):
                if num >= 1:
                    if self.auto_tuning:
                        self.logger.log_warning(f'Along the path {self.mask_directory} there are several filter cubes, the file will be selected as a basis - {filename}')
                        break
                    else:
                        self.logger.log_error(f'There are several region cubes along the path {self.mask_directory}. It is necessary to leave only one filter cube')
                        raise AssertionError(f'There are several region cubes along the path {self.mask_directory}. It is necessary to leave only one filter cube')
                file_path = os.path.join(self.mask_directory, filename)
                if os.path.isfile(file_path):
                    self.__file_mask_path = file_path
        else:
            self.logger.log_warning(f'There are no unloaded region property cubes at path {self.mask_directory}')

        # Читаем координатную сетку
        if os.path.exists(self.coord_directory):
            for num, filename in enumerate(os.listdir(self.coord_directory)):
                if num >= 1:
                    if self.auto_tuning:
                        self.logger.log_warning(f'There are several filter cubes along the path {self.coord_directory}, the file will be selected as a basis - {filename}')
                        break
                    else:
                        self.logger.log_error(f'There are several region cubes along the path {self.coord_directory}. It is necessary to leave only one filter cube')
                        raise AssertionError(f'There are several region cubes along the path {self.coord_directory}. It is necessary to leave only one filter cube')
                file_path = os.path.join(self.coord_directory, filename)
                if os.path.isfile(file_path):
                    self.__file_coord_path = file_path
        else:
            self.logger.log_error(f'There are no unloaded property cubes at path {self.coord_directory}')
            raise AssertionError(f'There are no unloaded property cubes at path {self.coord_directory}')

    def __output_files(self):
        """
        :oprion: Возращает куб cвойст в формате [nx, ny, nz]
        :return: np.ndarray
        """

        ''' Читаем выгзрузку свойств '''

        # region

        def __replace_99999(value):
            if value == 99999:
                return 0
            return value

        self.logger.log_info(f'Reading 3D mesh file')
        with open(self.__file_coord_path, 'r') as f:
            lines = f.readlines()
            self.x_n, self.y_n, self.z_n = map(int, lines[0].split())
            __contents_coords = np.array([list(map(float, line.split())) for line in lines[1:]])

        self.x_min = np.min(__contents_coords[:, 0])
        self.x_max = np.max(__contents_coords[:, 0])
        self.y_min = np.min(__contents_coords[:, 1])
        self.y_max = np.max(__contents_coords[:, 1])
        self.z_min = np.min(__contents_coords[:, 2])
        self.z_max = np.max(__contents_coords[:, 2])
        self.x_line = np.linspace(self.x_min, self.x_max, self.x_n)
        self.y_line = np.linspace(self.y_min, self.y_max, self.y_n)
        self.z_line = np.linspace(self.z_min, self.z_max, self.z_n)

        self.logger.log_info(f"Grid parameters by axis x: min = {self.x_min}, n = {self.x_n}, max = {self.x_max}")
        self.logger.log_info(f"Grid parameters by axis y: min = {self.y_min}, n = {self.y_n}, max = {self.y_max}")
        self.logger.log_info(f"Grid parameters by axis z: min = {self.z_min}, n = {self.z_n}, max = {self.z_max}")

        if __contents_coords.shape[0] != self.x_n * self.y_n * self.z_n:
            self.logger.log_error(f"Grid size = {__contents_coords.shape[0]} does not match с x_n * y_n * z_n = {self.x_n * self.y_n * self.z_n}")
            raise AssertionError(f"Grid size = {__contents_coords.shape[0]} does not match с x_n * y_n * z_n = {self.x_n * self.y_n * self.z_n}")

        __contents_property = {}
        for file_path, file_name in zip(self.__file_property_path, self.__file_property_names):
            self.logger.log_info(f'Reading a file {file_name}')
            __content_property = []
            with open(file_path, 'r') as __file:
                for _ in range(4):
                    next(__file)
                for __line in __file:
                    __elements = __line.strip().split()
                    for __element in __elements:
                        if "*" in __element:
                            __count, __value = __element.split('*')
                            __count = int(__count)
                            __content_property.extend([float(__value)] * __count)
                        elif __element == '/':
                            continue
                        else:
                            __content_property.append(float(__element))
            __content_property = np.array(__content_property).flatten()
            __contents_property[file_name] = __content_property

        # Куб регионов
        if self.__file_regions_path is not None:
            __contents_region = []
            self.logger.log_info(f'Reading a region cube file')
            with open(self.__file_regions_path, 'r') as __file:
                for _ in range(4):
                    next(__file)
                for __line in __file:
                    __elements = __line.strip().split()
                    for __element in __elements:
                        if "*" in __element:
                            __count, __value = __element.split('*')
                            __count = int(__count)
                            __contents_region.extend([float(__value)] * __count)
                        elif __element == '/':
                            continue
                        else:
                            __contents_region.append(float(__element))
            __contents_region = np.array(__contents_region).flatten()
        else:
            __contents_region = np.zeros(__content_property.shape)

        # Куб маски
        if self.__file_mask_path is not None:
            __contents_mask = []
            self.logger.log_info(f'Reading a filter cube file')
            with open(self.__file_mask_path, 'r') as __file:
                for _ in range(4):
                    next(__file)
                for __line in __file:
                    __elements = __line.strip().split()
                    for __element in __elements:
                        if "*" in __element:
                            __count, __value = __element.split('*')
                            __count = int(__count)
                            __contents_mask.extend([float(__value)] * __count)
                        elif __element == '/':
                            continue
                        else:
                            try:
                                pass
                            except Exception as e:
                                continue
                            __contents_mask.append(float(__element))
            __contents_mask = np.array(__contents_mask).flatten()
            if not np.isin(__contents_mask, [0, 1]).all():
                if self.auto_tuning:
                    self.logger.log_warning(f"The file with the filtering property must contain only 0 and/or 1. Ignore this property cube")
                    __contents_mask = np.ones(__contents_mask.shape)
                else:
                    self.logger.log_error(f"The file with the filter property must contain only 0 and/or 1")
                    raise AssertionError("The file with the filter property must contain only 0 and/or 1")

        __combined_grid = np.stack(list(__contents_property.values()), axis=0)
        __contents_grid = np.prod(__combined_grid, axis=0)
        if self.__file_mask_path is not None:
            __contents_grid = __contents_grid * __contents_mask

        self.grid_cube = __contents_grid
        self.region_cube = __contents_region
        self.coords = __contents_coords

    def __call__(self, *args, **kwargs):
        Direct.__call__(self)
        self.__read_files()
        self.__output_files()

    @property
    def get_coords(self) -> np.ndarray:
        """
        :options:
        Трехмерный массив всех точек с столбцами X, Y, Z \n
        Количество элементов совпадает с атрибутом values

        :return: np.ndarray(np.ndarray, np.ndarray, np.ndarray)
        """
        return self.coords

    @property
    def get_values(self) -> np.ndarray:
        """
        :option:
        Одномерный массив всех точек \n
        Количество эклментов совпадает с атрибутом coords

        :return: np.ndarray
        """
        return self.grid_cube

    @property
    def get_regions(self) -> np.ndarray:
        """
        :option:
        Одномерный массив всех точек \n
        Количество эклментов совпадает с атрибутом coords

        :return: np.ndarray
        """
        return self.region_cube

    def get_current_value(self, coord: np.ndarray) -> np.ndarray:
        """
        Функция для получения значения свойства по заданным координатам
        :param coord: координаты точки в виде (x, y, z)
        :return: значение свойства в заданной точке
        """
        x, y, z = coord
        i = np.searchsorted(self.x_line, x)
        j = np.searchsorted(self.y_line, y)
        k = np.searchsorted(self.z_line, z)

        # Проверка на выход за пределы сетки
        if i == 0 or i == len(self.x_line) or j == 0 or j == len(self.y_line) or k == 0 or k == len(self.z_line):
            # Обработка краевых точек
            if i == len(self.x_line):
                i -= 1
            if j == len(self.y_line):
                j -= 1
            if k == len(self.z_line):
                k -= 1

        # Корректировка индексов, если координата точно совпадает с границей сетки
        if x == self.x_line[i]:
            i -= 1
        if y == self.y_line[j]:
            j -= 1
        if z == self.z_line[k]:
            k -= 1

        index = i * (self.y_n * self.z_n) + j * self.z_n + k
        return self.get_values[index]

    def get_current_values(self, coords: np.ndarray) -> np.ndarray:
        """
        Функция для получения значений свойства по заданным координатам
        :param coords: массив координат точек в виде (x, y, z)
        :return: массив значений свойства в заданных точках
        """
        values = np.zeros(len(coords))
        for idx, coord in enumerate(coords):
            values[idx] = self.get_current_value(coord)
        return values