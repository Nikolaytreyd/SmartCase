import json
from datetime import datetime
import shutil
import os

class Manager:
    def __init__(self, variable_file: str, config_varibale: str, default_variables: str):
        self.variable_file = Manager.check_file(variable_file)
        self.config_variable_file = Manager.check_file(config_varibale)
        self.default_variables_file = Manager.check_file(default_variables)
        self.load_variable(), self.load_config_variable()

    @staticmethod
    def check_file(file_path: str):
        """
        Проверяет, что файл существует, является JSON файлом и другие условия.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден.")
        if not file_path.lower().endswith('.json'):
            raise ValueError(f"Файл {file_path} не является JSON файлом.")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Файл {file_path} не доступен для чтения.")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Файл {file_path} пустой.")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Файл {file_path} содержит некорректный JSON.")
        return file_path

    @staticmethod
    def convert_value_from_str(value, variable_type):
        """
        Преобразует значение в соответствующий тип данных.
        :param value: Значение переменной.
        :param variable_type: Тип данных переменной.
        :return: Преобразованное значение.
        """
        if value == '':
            return None
        if variable_type == "int":
            return int(value)
        elif variable_type == "float":
            return float(value)
        elif variable_type == "list":
            return json.loads(value)
        elif variable_type == "datetime":
            return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            # return datetime.fromisoformat(value)
        elif variable_type == 'bool':
            return bool(value)
        else:
            return value

    @staticmethod
    def convert_value_to_str(value, variable_type):
        """
        Преобразует значение в строку для сохранения в JSON.
        :param value: Значение переменной.
        :param variable_type: Тип данных переменной.
        :return: Преобразованное значение в строку.
        """
        if variable_type == "list":
            return json.dumps(value)
        elif variable_type == "datetime":
            return value.strftime('%Y-%m-%d %H:%M:%S')
            # return value.isoformat()
        else:
            return str(value)

    def load_variable(self):
        """
        Чтение json файла со значениями переменных
        :return: None
        """
        with open(self.variable_file, 'r', encoding='utf-8') as file:
            variable = json.load(file)
        self.variable = variable

    def load_config_variable(self):
        """
        Чтение json файла с настройками переменными
        :return: None
        """
        with open(self.config_variable_file, 'r', encoding='utf-8') as file:
            config_variable = json.load(file)
        self.config_variable = config_variable

    @property
    def info_variable(self):
        """
        :return: Dict со значениями переменных
        """
        return self.variable

    @property
    def info_variable_settings(self):
        """
        :return: Dict настроек переменных
        """
        return self.config_variable

    def get_variable(self, key: str):
        """
        Получает значение переменной по указанным ключам.
        :param key: Ключ для доступа к переменной.
        :return: Значение переменной.
        """
        if key in self.variable:
            if isinstance(key, str):
                return Manager.convert_value_from_str(self.variable[key], self.config_variable[key]['type'])
            else:
                raise AssertionError("ключ {key} к переменной должен задаваться через ''")
        else:
            raise AssertionError(f"Переменная '{key}' не существует")

    def set_variable(self, key: str, value):
        """
        Устанавливает значение переменной по указанным ключам.
        :param key: Ключи для доступа к переменной.
        :param value: Новое значение переменной.
        """
        if key in self.variable:
            if isinstance(key, str):
                self.variable[key] = Manager.convert_value_to_str(value, self.config_variable[key]['type'])
                self.save_variable()
            else:
                raise AssertionError("ключ {key} к переменной должен задаваться через ''")
        else:
            raise AssertionError(f"Переменная '{key}' не существует")

    def save_variable(self):
        """
        Сохраняет изменения в JSON файл.
        """
        with open(self.variable_file, 'w', encoding='utf-8') as file:
            json.dump(self.variable, file, ensure_ascii=False, indent=4)

    def help_variable(self, variable_name):
        """
        Получает информацию о переменной по её имени.
        :param variable_name: Имя переменной.
        :return: Информация о переменной.
        """
        variable_info = self.config_variable.get(variable_name, "Переменная не найдена")
        if isinstance(variable_info, dict):
            print(f"Информация о переменной '{variable_name}':")
            for key, value in variable_info.items():
                print(f" {key}: {value}")
        else:
            print(variable_info)
        return variable_info

    def help_field(self, field_name):
        """
        Получает информацию о всех переменных по указанному полю.
        :param field_name: Имя поля (например, 'description').
        :return: Словарь с информацией о переменных.
        """
        field_info = {}
        for variable_name, variable_info in self.config_variable.items():
            if field_name in variable_info:
                field_info[variable_name] = variable_info[field_name]
                print(f"{variable_name}: {variable_info[field_name]}")
        return field_info

    def reset_to_default(self):
        """
        Сбрасывает все изменения и возвращает переменные к значениям по умолчанию.
        """
        shutil.copyfile(self.default_variables_file, self.variable_file)
        self.load_variable()


manager_oil = Manager(
    variable_file=r"D:\Orher\oil\variable.json",
    config_varibale=r"D:\14. Кейсы\005. IRM\models\Example\02_oil\variable_directory\conditions_variable.json",
    default_variables=r"D:\14. Кейсы\005. IRM\models\Example\02_oil\variable_directory\by_default_variables.json"
)

print(manager_oil.get_variable('start'))
manager_oil.set_variable('start', datetime(year=2024, month=1, day=1))
print(manager_oil.get_variable('start'))

print(manager_oil.help_field('description'))
print(manager_oil.help_field('type'))