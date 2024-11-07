from config.importation import *
from input.load_json import Load
from config.log import Logger
from config.config import global_random

class Variable (Load):
    def __init__(self, variable_file):
        Load.__init__(self, variable_file)
        self.attributes = {}
        self.logger = Logger(enable_file_logging=True)

    def define_attributes(self):
        for key, value in self.variable.items():
            validated_value = self.validate_variable(key, self.convert_value_from_str(value, self.config_variable[key]['type']))
            setattr(self, key, validated_value)
            self.logger.log_info(f'Variable | {key} = {validated_value} | defined in the attributes of the calculation class')
            str_value = self.convert_value_to_str(validated_value, self.config_variable[key]['type'])
            self.attributes[key] = str_value

    def validate_variable(self, key, value):
        """
        Проверяет тип данных, ограничения и корректность значения переменной.
        :param key: Ключ переменной.
        :param value: Значение переменной.
        :return: Проверенное значение переменной.
        """
        variable_info = self.config_variable[key]
        variable_type = self.config_variable[key]['type']
        status_empty = variable_info.get("status_empty", False)

        if value is None or value == "":
            if status_empty:
                self.logger.log_error(f"Variable '{key}' can't be empty")
                raise AssertionError(f"Variable '{key}' can't be empty")
            else:
                return None

        if key not in self.config_variable:
            self.logger.log_error(f"Variable '{key}' does not exist")
            raise AssertionError(f"Variable '{key}' does not exist")

        if not isinstance(value, eval(variable_type)):
            self.logger.log_error(f"Invalid data type for variable '{key}'. Expected '{variable_type}', received'{type(value).__name__}'")
            raise AssertionError(f"Invalid data type for variable '{key}'. Expected '{variable_type}', received '{type(value).__name__}'")

        if variable_type in ["int", "float"]:
            if variable_info.get("limit", {}).get("min") == '-np.inf':
                min_value = -np.inf
            else:
                min_value = self.convert_value_from_str(variable_info.get("limit", {}).get("min"), self.config_variable[key]['type'])
            if variable_info.get("limit", {}).get("max") == 'np.inf':
                max_value = np.inf
            else:
                max_value = self.convert_value_from_str(variable_info.get("limit", {}).get("max"), self.config_variable[key]['type'])
            if self.get_variable('auto_tuning'):
                if min_value is not None and value < min_value:
                    self.logger.log_warning(f"Variable value '{key}' less than minimum {min_value} < {value}. The threshold minimum value has been set")
                    value = min_value
                if max_value is not None and value > max_value:
                    self.logger.log_warning(f"Variable value '{key}' more than maximum {max_value} > {value}. The threshold maximum value has been set")
                    value = max_value
            else:
                if min_value is not None and value < min_value:
                    self.logger.log_error(f"Variable value '{key}' less than minimum {min_value}.")
                    raise ValueError(f"Variable value '{key}' less than minimum {min_value}.")
                if max_value is not None and value > max_value:
                    self.logger.log_error(f"Variable value '{key}' more than maximum  {max_value}.")
                    raise ValueError(f"Variable value '{key}' more than maximum  {max_value}.")

        # Проверка корректности значения для переменных, заканчивающихся на _directory
        if key.endswith("_directory"):
            if not isinstance(value, str):
                self.logger.log_error(f"{key} Variable value {key} must be a string")
                raise AssertionError(f"{key} Variable value {key} must be a string")
            if not os.path.exists(value) or not os.path.isdir(value):
                self.logger.log_error(f"{key} Path '{value}' does not exist or is not a directory")
                raise AssertionError(f"{key} Path '{value}' does not exist or is not a directory")
                # Проверка наличия файлов с определенными форматами
            if key in ["property_directory", "regions_directory", "mask_directory", "coord_directory"]:
                files = os.listdir(value)
                if key == "property_directory":
                    if not any(file.lower().endswith(('.inc', '.INC', '.Inc')) for file in files):
                        self.logger.log_error(f"{key} In the folder '{value}' no files found with extension <.inc>, <.INC> or <.Inc>")
                        raise AssertionError(f"{key} In the folder '{value}' no files found with extension <.inc>, <.INC> or <.Inc>")
                else:
                    if files:
                        if not any(file.lower().endswith(('.inc', '.INC', '.Inc')) for file in files):
                            self.logger.log_error(f"{key} In the folder '{value}' no files found with extension <.inc>, <.INC> or <.Inc>")
                            raise AssertionError(f"{key} In the folder'{value}' no files found with extension <.inc>, <.INC> or <.Inc>")
        # Проверка корректности значения для переменной map_directory
        if key == "map_directory":
            if not isinstance(value, str):
                self.logger.log_error("Variable value 'map_directory' must be a string")
                raise AssertionError("Variable value 'map_directory' must be a string")
            files = os.listdir(value)
            if len(files) > 1:
                if self.get_variable('auto_tuning'):
                    self.logger.log_warning(f"Warning: In the folder  '{value}' more than one file found. The first file will be used.")
                else:
                    self.logger.log_error(f"{key} In the folder  '{value}' more than one file found")
                    raise AssertionError(f"{key} In the folder '{value}' more than one file found")
            files = os.listdir(value)
            if not any(file.lower().endswith(('.png', '.PNG', '.jpeg', '.JPEG')) for file in files):
                self.logger.log_error(f"{key} In the folder '{value}' no files found with extension <.png> or <.jpeg>")
                raise AssertionError(f"In the folder '{value}' no files found with extension <.png> or <.jpeg>")

        # Проверка корректности значения для переменной name
        if key == "name":
            if not isinstance(value, str):
                self.logger.log_error(f"The calculation project {key}  name must be string '{value}'")
                raise AssertionError(f"The calculation project {key} name must be string '{value}'")
            if re.search(r'[<>:"/\\|?*]', value):
                self.logger.log_error(f"Invalid characters in {key} value of the variable '{value}'")
                raise AssertionError(f"Invalid characters in {key} value of the variable'{value}'")

        # Проверка корректности значения для переменных rgb_lower и rgb_upper
        if key in ["rgb_lower", "rgb_upper"]:
            if not isinstance(value, list):
                self.logger.log_error(f"Variable value {key} = {value} должно быть str(list) = '[[value1, value2, value3], ... ]' ")
                raise AssertionError(f"Variable value {key} = {value} должно быть str(list) = '[[value1, value2, value3], ... ]' ")
            for index, v in enumerate(value):
                if isinstance(v, int):
                    continue
                if not isinstance(v, list):
                    self.logger.log_error(f"Variable value {key} = {value}: {index} = {v}, must be an array list")
                    raise AssertionError(f"Variable value {key} = {value}: {index} = {v}, must be an array list")
                if not len(v) == 3:
                    self.logger.log_error(f"Variable value {key} = {value}: {index} = {v}, must have three elements [value1=R, value2=G, value3=B]")
                    raise AssertionError(f"Variable value {key} = {value}: {index} = {v}, must have three elements [value1=R, value2=G, value3=B]")
                for color, m in enumerate(v):
                    if m < 0 or m > 255:
                        self.logger.log_error(f"Variable value {key} = {value}: list[{index}][{color}] = {m}, must have a variation: [0; 255]")
                        raise AssertionError(f"Variable value {key} = {value}: list[{index}][{color}] = {m}, must have a variation: [0; 255]")
            for i, bound in enumerate(value):
                if isinstance(bound, int):
                    continue
                if i != 0:
                    if not (
                            value[i - 1][0] <= value[i][0]
                            and value[i - 1][1] <= value[i][1]
                            and value[i - 1][2] <= value[i][2]
                    ):
                        self.logger.log_error(f"Variable value {key} = {value}: RGB boundaries should be filled in ascending order")
                        raise AssertionError(f"Variable value {key} = {value}: RGB boundaries should be filled in ascending order")

        return value

    def validate_between_variables(self):
        # Проверка мин/макс углов, длины
        if self.length_min >= self.length_max:
            self.logger.log_error(f"The specified minimum barrel length is greater than the specified maximum barrel length"
                                  f"length_min >= lenght_max = {self.length_min} >= {self.length_max}")
            raise AssertionError(f"The specified minimum barrel length is greater than the specified maximum barrel length:"
                                  f"length_min >= lenght_max = {self.length_min} >= {self.length_max}")
        delta_lenght = self.length_max - self.length_min
        if delta_lenght <= 500:
            self.logger.log_warning(f"Предупреждение малого коридора допустимых длин стволов скважин "
                                    f"lenght_max  - lenght_min = {self.length_max} - {self.length_min} = {delta_lenght} м")
        if self.angle_min >= self.angle_max:
            self.logger.log_error(f"The specified minimum cutting angle of the table in the GS is greater than the specified maximum cutting angle of the trunk in the GS "
                                  f"angle_min >= angle_max = {self.angle_min} >= {self.angle_max}")
            raise AssertionError(f"The specified minimum cutting angle of the table in the GS is greater than the specified maximum cutting angle of the trunk in the GS "
                                  f"angle_min >= angle_max = {self.angle_min} >= {self.angle_max}")
        delta_angle = self.angle_max - self.angle_min
        if delta_angle <= 20:
            self.logger.log_warning(f"Warning of a small corridor of permissible angles of cutting of wellbore trunks in relation to the horizontal wellbore "
                                    f"angle_max  - angle_min = {self.angle_max} - {self.angle_min} = {delta_angle} град.")

        # Нужно добавить условия заполнения map данных. Если есть ссылка на карту, то другие перменные должны быть

    def __call__(self):
        self.define_attributes()
        self.validate_between_variables()
        if self.global_random_int == 0:
            global_random(self.global_random_int)
            self.logger.log_warning(f'Initialization of the reproducible calculation: seed - {self.global_random_int}')
        else:
            self.logger.log_warning(f'Initialization of non-reproducible calculation')
