#  Файл создан tNavigator v21.3-2272-g351282a5db0.
#  Copyright (C) RFDynamics 2005-2021.
#  Все права защищены.

# This file is MACHINE GENERATED! Do not edit.

#api_version=v0.0.36

from __main__.tnav.workflow import *
from tnav_debug_utilities import *
from datetime import datetime, timedelta


declare_workflow (workflow_name="template",
      variables=[{"name" : "SELECTION_OIL", "type" : "real", "min" : 20, "max" : 90, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [20, 30, 40, 50, 60, 70, 80, 90], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "SELECTION_GAS", "type" : "real", "min" : 20, "max" : 90, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [20, 30, 40, 50, 60, 70, 80, 90], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WP_MAX_OIL", "type" : "real", "min" : 5, "max" : 12, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [5, 6, 7, 8, 9, 10, 11, 12], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WP_MAX_GAS", "type" : "real", "min" : 5, "max" : 12, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [5, 6, 7, 8, 9, 10, 11, 12], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WT_MAX_OIL", "type" : "real", "min" : 1, "max" : 4, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3, 4], "discrete_distr_probabilities" : [25, 25, 25, 25], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WT_MAX_GAS", "type" : "real", "min" : 1, "max" : 4, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3, 4], "discrete_distr_probabilities" : [25, 25, 25, 25], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "MOBIL_BU_OIL", "type" : "real", "min" : 1, "max" : 3, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3], "discrete_distr_probabilities" : [33.34, 33.33, 33.33], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "MOBIL_BU_GAS", "type" : "real", "min" : 1, "max" : 3, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3], "discrete_distr_probabilities" : [33.34, 33.33, 33.33], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DEP_GAS", "type" : "real", "min" : 2, "max" : 45, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [2, 5, 10, 15, 20, 25, 30, 35, 40, 45], "discrete_distr_probabilities" : [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DEP_OIL", "type" : "real", "min" : 2, "max" : 45, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [2, 5, 10, 15, 20, 25, 30, 35, 40, 45], "discrete_distr_probabilities" : [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}])


template_variables = {
"SELECTION_OIL" : 70,
"SELECTION_GAS" : 70,
"WP_MAX_OIL" : 5,
"WP_MAX_GAS" : 5,
"WT_MAX_OIL" : 4,
"WT_MAX_GAS" : 2,
"MOBIL_BU_OIL" : 1,
"MOBIL_BU_GAS" : 1,
"DEP_GAS" : 2,
"DEP_OIL" : 2
}

def template (variables = template_variables):
    pass
    check_launch_method ()

    SELECTION_OIL = variables["SELECTION_OIL"]
    SELECTION_GAS = variables["SELECTION_GAS"]
    WP_MAX_OIL = variables["WP_MAX_OIL"]
    WP_MAX_GAS = variables["WP_MAX_GAS"]
    WT_MAX_OIL = variables["WT_MAX_OIL"]
    WT_MAX_GAS = variables["WT_MAX_GAS"]
    MOBIL_BU_OIL = variables["MOBIL_BU_OIL"]
    MOBIL_BU_GAS = variables["MOBIL_BU_GAS"]
    DEP_GAS = variables["DEP_GAS"]
    DEP_OIL = variables["DEP_OIL"]

    begin_user_imports ()
    import numpy as np
    import json
    import shutil
    import os
    end_user_imports ()

    if False:
        begin_wf_item (index = 1, is_custom_code = True, name = "Библиотеки")
        exec ("""
# Библиотеки здесь, если вдруг потеряются
#import numpy as np
#import json
#import shutil
#import os
#import re
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter
#from scipy.ndimage import median_filter
""")
        end_wf_item (index = 1)


    begin_wf_item (index = 2)
    comment_text ("""
Переменные которые принимают абсолютный путь (без русских букв) к папкам (директориям) и файлам json

Необходимо создать папки с именами (даже если по ним не будут файлы):

	Директории которые ОБЯЗАТЕЛЬНО должны иметь внтури файлы:
	1) coord_direcotory - абсолютный путь к директории, где будет сохранена сетка 
	Важно: сетка должна иметь пропорциональное разбиение.
	Важно: Все блоки активные
	2) property_directory - абсолютный путь к директории, где будет сохранены кубы свойств
	3) project_directory - абсолютный путь к результам расчета скрипта

	Директории которые НЕ ОБЯЗАТЕЛЬНО должны иметь внтури файлы:
	4) mask_directory - абсолютный путь к директории где выгружен бинарный куб (0 и 1) для фильтрации. Если нет такого куба указываем \"\"
	5) region_directory - абсолютный путь к директории где выгружен дискретный куб блоков или регионов. Если нет такого куба указываем \"\"
	6) map_directory - абсолютный путь к директории где есть растровая карта поврехности

Необходимо указать пути к файлам:
	Важно: путь к файлу должен указываться как = r\"direct1/direct2/file.json\"\" или \"direct1//direct2//file.json\"

	Можно менять файл:
	1) variable_file - Заполняемый шаблон значений переменных.  Будет использован в качестве переменной для запуска main.exe

	Менять не рекомендуется, т.к. SmartCase.exe ссылается своему config_varibale и default_variables:
	2) default_variables - Пустой шаблон для сброса изменений значений переменных установленных в variable_file
	3) config_varibale - json файл с описанием переменных, типы, ограничения 


grid_name = Название текущей сетки
""")
    end_wf_item (index = 2)


    begin_wf_item (index = 3, name = "Пути к директориям")
    grid_name = "test_grid"
    set_var_type (n = "grid_name", t = "STRING", it = "PY_EXPR", val = grid_name)
    variables["GRID_NAME"] = grid_name
    coord_directory = r"D:\Orher\coord_directory"
    set_var_type (n = "coord_directory", t = "STRING", it = "PY_EXPR", val = coord_directory)
    variables["COORD_DIRECTORY"] = coord_directory
    property_directory = r"D:\Orher\property_directory"
    set_var_type (n = "property_directory", t = "STRING", it = "PY_EXPR", val = property_directory)
    variables["PROPERTY_DIRECTORY"] = property_directory
    project_directory = r"D:\Orher\project_directory"
    set_var_type (n = "project_directory", t = "STRING", it = "PY_EXPR", val = project_directory)
    variables["PROJECT_DIRECTORY"] = project_directory
    region_directory = r"D:\Orher\regions_directory"
    set_var_type (n = "region_directory", t = "STRING", it = "PY_EXPR", val = region_directory)
    variables["REGION_DIRECTORY"] = region_directory
    mask_directory = r"D:\Orher\mask_directory"
    set_var_type (n = "mask_directory", t = "STRING", it = "PY_EXPR", val = mask_directory)
    variables["MASK_DIRECTORY"] = mask_directory
    map_directory = r"D:\Orher\map_directory"
    set_var_type (n = "map_directory", t = "STRING", it = "PY_EXPR", val = map_directory)
    variables["MAP_DIRECTORY"] = map_directory

    end_wf_item (index = 3)


    begin_wf_item (index = 4, name = "Пути к файлам json")
    variable_file = r"D:\\Orher\\oil\\variable.json"
    set_var_type (n = "variable_file", t = "STRING", it = "PY_EXPR", val = variable_file)
    variables["VARIABLE_FILE"] = variable_file
    config_varibale = r"D:\Orher\variable_directory\conditions_variable.json"
    set_var_type (n = "config_varibale", t = "STRING", it = "PY_EXPR", val = config_varibale)
    variables["CONFIG_VARIBALE"] = config_varibale
    default_variables = r"D:\Orher\variable_directory\by_default_variables.json"
    set_var_type (n = "default_variables", t = "STRING", it = "PY_EXPR", val = default_variables)
    variables["DEFAULT_VARIABLES"] = default_variables

    end_wf_item (index = 4)


    if False:
        begin_wf_item (index = 5, name = "Сохранение сетки @grid_name@")
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 6)
                grid_property_calculator (mesh=find_object (name=resolve_variables_in_string (string_with_variables="@GRID_NAME@",
                      variables=variables),
                      type="Grid3d"),
                      result_grid_property=find_object (name="X_",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="zone_id",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="X",
                      variables=variables)
                end_wf_item (index = 6)


            if False:
                begin_wf_item (index = 7)
                grid_property_calculator (mesh=find_object (name=resolve_variables_in_string (string_with_variables="@GRID_NAME@",
                      variables=variables),
                      type="Grid3d"),
                      result_grid_property=find_object (name="Y_",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="zone_id",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="Y",
                      variables=variables)
                end_wf_item (index = 7)


            if False:
                begin_wf_item (index = 8)
                grid_property_calculator (mesh=find_object (name=resolve_variables_in_string (string_with_variables="@GRID_NAME@",
                      variables=variables),
                      type="Grid3d"),
                      result_grid_property=find_object (name="Z_",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="zone_id",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="Z",
                      variables=variables)
                end_wf_item (index = 8)


            if False:
                begin_wf_item (index = 9, is_custom_code = True, name = "Выгрузка сетки")
                g = get_grid_by_name(name=grid_name)
                property_x_ = g.get_property_by_name(name='X_')
                property_y_ = g.get_property_by_name(name='Y_')
                property_z_ = g.get_property_by_name(name='Z_')

                x = property_x_.get_np_array()
                y = property_y_.get_np_array()
                z = property_z_.get_np_array()

                coord = np.zeros((x.size, 3))
                index = 0
                for k in range(x.shape[2]):
                	print(f"Чтение слоя {k+1} из {x.shape[2]}")
                	for j in range(x.shape[1]):
                		for i in range(x.shape[0]):
                			coord[index, :] = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                			index += 1

                file_path = f"{coord_directory}\\coords.INC"
                with open(file_path, 'w') as f:
                	f.write(f"{x.shape[0]} {x.shape[1]} {x.shape[2]}\n")
                	for row in coord:
                		f.write(' '.join(map(str, row)) + '\n')

                print(f"{file_path} сетка сохранена")
                end_wf_item (index = 9)



        end_wf_item (index = 5)


    if False:
        begin_wf_item (index = 11, name = "Сохранение кубов свойств")
        workflow_folder ()
        if True:
            pass



            begin_wf_item (index = 12)
            comment_text ("""
1) Определяем правильно имя сетки
2) Файлы кубов свойств сохраняем в директории соглано @property_directory@ (можно сохранить сразу несколько кубов)
3) Файл куба региона сохраняем в директории @region_directory@ (только один файл, иначе будет выбран только первый файл в директории @region_directory@)
4) Файл куба маски сохраняем в директории @mask_directory@ (только один файл, иначе будет выбран только первый файл в директории @mask_directory@)
""")
            end_wf_item (index = 12)


            if False:
                begin_wf_item (index = 13, name = "нефтенасыщенность")
                grid_property_export_gridecl_format (grid=find_object (name="test_grid",
                      type="Grid3d"),
                      grid_property=find_object (name="So__init",
                      type="Grid3dProperty"),
                      file_name="../../../maps/propertys/SO__INIT.INC",
                      use_precision=False,
                      precision=0,
                      keyword="SO_INIT",
                      inactive_placeholder="0",
                      separate_by_comment=False,
                      units_system="metric")
                end_wf_item (index = 13)


            if False:
                begin_wf_item (index = 14, name = "проницаемость")
                grid_property_export_gridecl_format (grid=find_object (name="test_grid",
                      type="Grid3d"),
                      grid_property=find_object (name="PERMX",
                      type="Grid3dProperty"),
                      file_name="../../../maps/propertys/PERMX.INC",
                      use_precision=False,
                      precision=0,
                      keyword="PERMX",
                      inactive_placeholder="0",
                      separate_by_comment=False,
                      units_system="metric")
                end_wf_item (index = 14)



        end_wf_item (index = 11)


    if False:
        begin_wf_item (index = 16, is_custom_code = True, name = "Manager param")

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
                    # Если Python имеет версию ранее 3.7
                    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                    # Если Python имеет версию позже 3.7
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
                    # Если Python имеет версию ранее 3.7
                    return value.strftime('%Y-%m-%d %H:%M:%S')
                    # Если Python имеет версию позже 3.7
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
        end_wf_item (index = 16)


    if False:
        begin_wf_item (index = 17, is_custom_code = True, name = "Определение переменных моделей")
        manager_oil = Manager(
        	variable_file = variable_file, 
        	config_varibale = config_varibale, 
        	default_variables = default_variables
        	)

        # Вывод дополнительной информации о характеристик переменных
        print(manager_oil.help_field('description'))
        #		Доступно 
        #		"type", Тип данных
        #		"measure", Едицина измерения
        #		"description", Описание переменной
        #		"status_empty", Статус переменной, если переменная моежт быть пустой - True,иначе False
        #		"status_variable", Статус переменной, для чего используется переменная
        #		"limit" Верхние и нижние границы диапазаонов вариации переменных

        '''

        			Описание принимаемых атрибутов класса Manager

        variable_file = Заполняемый шаблон значений переменных. Который будет использован в качестве переменной для запуска main.exe
        config_varibale = Json файл с описанием переменных, типы, ограничения (менять не рекомендуется, т.к. main.exe ссылается своему config_varibale)
        default_variables = Пустой шаблон для сброса изменений значений переменных установленных в variable_file

        '''


        def get(key:str) -> None:
        	'''
        	Вывод сообщения 
        	'''
        	print(f"Установленное значение {key} = {manager_oil.get_variable(key)}")


        '''

        			Обязательные параметры расчета

        '''

        # Устанавливаем название расчета
        manager_oil.set_variable('name', 'OIL')
        get('name')

        # Устанавливаем начало расчета (согласно начальному времени расчета в проекте Дизайнера моеделей)
        manager_oil.set_variable('start', datetime(year=2024, month=1, day=1))
        get('start')

        # Устанавливаем время расчета (согласно времени расчета в проекте Дизайнера моеделей, в годах)
        manager_oil.set_variable('duration', 20)
        get('duration')
        #
        # Устанавливаем абсолютный путь к выгруженным кубам свойств. В данном случае к директории с PERMX, OIPM
        manager_oil.set_variable('property_directory', property_directory)
        get('property_directory')

        # Устанавливаем абсолютный путь к выгрузке координатной сетки
        manager_oil.set_variable('coord_directory', coord_directory)
        get('coord_directory')

        # Устанавливаем абсолютный путь для сохранения результатов расчета скрипта
        manager_oil.set_variable('project_directory', project_directory)
        get('project_directory')

        '''

        			Необязательные параметры

        '''

        # Устанавливаем абсолютный путь к выгруженныму кубу регионов. В данном случае к директории с EQLNUM
        manager_oil.set_variable('regions_directory', region_directory)
        get('regions_directory')

        # Устанавливаем абсолютный путь к выгруженныму кубу фильтра. В данном случае к директории, которая пустая (опция расчета с фильтром не будет использована)
        manager_oil.set_variable('mask_directory', mask_directory)
        get('mask_directory')

        # Устанавливаем путь к растровой карте поверхности.
        # ВАЖНО!: Путь не должен содержать русскоязычные символы (из-за особенностей работы cv2)
        manager_oil.set_variable('map_directory', map_directory)

        # Дневная задержка по мобилизации БУ от начала времени расчета
        manager_oil.set_variable('relative_mob_start', 0)
        get('relative_mob_start')

        # Дневная задержка по запуску кейса от начала времени расчета
        # Допускаем, что от начала расчета и с момента готовности нефтяной инфраструктуры должно пройти 1.5 лет
        time_pause = int(1.5*365)
        manager_oil.set_variable('relative_pmr_start', time_pause)
        get('relative_pmr_start')


        '''

        			Параметры оптимизации 

        '''

        # Устанавливаем параметр фильтрации для нормализованного куба PERMX x OIPM (Считается внутри main.exe)
        manager_oil.set_variable('selection', int(SELECTION_OIL))
        get('selection')

        # Устанавливаем ограничение на максимальное количество КП 
        manager_oil.set_variable('wp_max', int(WP_MAX_OIL))
        get('wp_max')

        # Устанавливаем ограничение на максимальное количество стволов 
        # wt_max - 1: заканчивание ГС, wt_max - 2: заканчивание двухстволка, wt_max = 3.заканчивание МЗС
        manager_oil.set_variable('wt_max', int(WT_MAX_OIL))
        get('wt_max')

        # Устанавливаем количество БУ под мобилиацию
        manager_oil.set_variable('mobil_dr', int(MOBIL_BU_OIL))
        get('mobil_dr')

        # Вывод всех переменных
        print(manager_oil.info_variable)

        # Вывод всех характеристик переменных
        print(manager_oil.info_variable_settings)
        end_wf_item (index = 17)


    if False:
        begin_wf_item (index = 18, name = "Запуск SmartCase.exe")
        execute_external_program (program="D:/14. Кейсы/005. IRM/res/SmartCase.exe",
              arguments="\"D:\\\\Orher\\\\oil\\\\variable.json\"",
              use_working_directory=False,
              working_directory="")
        end_wf_item (index = 18)


    begin_wf_item (index = 19, is_custom_code = True, name = "Удаление скважин")
    print(get_all_wells())
    for w in get_all_wells():
    	delete_well (name=w.name)
    end_wf_item (index = 19)


    begin_wf_item (index = 20, name = "ИЛИ")
    workflow_folder ()
    if True:
        pass



        begin_wf_item (index = 21)
        comment_text ("""
необходимо корректно определить директорию внтутри скрипта directory//current//welltrac.INC
(если нет папки current, оно появится & актуализируется после запуска SmartCase.exe)
""")
        end_wf_item (index = 21)


        if False:
            begin_wf_item (index = 22)
            wells_import_welltrack_format (wells=find_object (name="Wells",
                  type="gt_wells_entity"),
                  trajectories=find_object (name="Trajectories",
                  type="Trajectories"),
                  do_remove_existing_wells=False,
                  well_searcher="name",
                  splitter=True,
                  file_names=[],
                  splitter2=True,
                  use_oem_encoding=False,
                  add_zero_point=False,
                  invert_z=False,
                  use_well_filter=False,
                  result_well_filter=find_object (name="Well Filter 1",
                  type="WellFilter"),
                  file_datum_info=CrsInfo (crs_type="not_specified",
                  crs_code=None,
                  crs_name="",
                  crs_proj_string=None,
                  datum_name=None,
                  datum_bounds_inited=False,
                  datum_bounds_min_x=0,
                  datum_bounds_max_x=0,
                  datum_bounds_min_y=0,
                  datum_bounds_max_y=0,
                  datum_is_in_proj4=False),
                  xy_units_system="si",
                  z_units_system="si",
                  use_xy_units=True,
                  xy_units="metres",
                  use_z_units=True,
                  z_units="metres")
            end_wf_item (index = 22)


        begin_wf_item (index = 23)
        comment_text ("""
необходимо корректно определить директорию внтутри скрипта directory
(если нет папки current, оно появится & актуализируется после запуска SmartCase.exe)
""")
        end_wf_item (index = 23)


        begin_wf_item (index = 24, is_custom_code = True, name = "Определение траектории скважин")
        file_path = f'{project_directory}//current//welltrac.INC'
        with open(file_path, 'r') as file:
        	lines = file.readlines()

        well_name = None
        trajectory_table = []


        for l, line in enumerate(lines):
        	print(f"Чтение строки {l+1} из {len(lines)}")
        	data = line.split()
        	if not data == []:
        		if data[0] in ["welltrack", "Welltrack", "WELLTRACK"]:
        			if well_name is not None:
        				wells_create(well_name=well_name, trajectory_table=trajectory_table)
        				well_name = data[-1][1:-1]
        				trajectory_table = []
        			else:
        				well_name = data[-1][1:-1]
        		elif data == ["/"]:
        			continue
        		else:
        			md = float(data[-1])
        			x = float(data[0])
        			y = float(data[1])
        			z = float(data[2])
        			trajectory_table.append({'md': md, 'x': x, 'y': y, 'z': z})


        end_wf_item (index = 24)



    end_wf_item (index = 20)


    begin_wf_item (index = 26)
    schedule_rule_well_structure (schedule_strategy=find_object (name="Case",
          type="gt_schedule_rules_data"),
          use_rule_name=True,
          rule_name="Ввод конструкции скважин",
          object_set_type="all_objects",
          object_set_name="Все скважины",
          params_table=[{"table_name" : find_object (name="Конструкция скважин",
          type="gt_wells_events_data")}])
    end_wf_item (index = 26)


    begin_wf_item (index = 27, name = "ИЛИ")
    workflow_folder ()
    if True:
        pass



        begin_wf_item (index = 28)
        comment_text ("""
необходимо корректно определить директорию внтутри скрипта directory
""")
        end_wf_item (index = 28)


        if False:
            begin_wf_item (index = 29)
            schedule_import (reload_all=True,
                  file_name="Example/02_oil/project_directory/current/schedule.sch",
                  strategy=find_object (name="Case",
                  type="gt_schedule_rules_data"),
                  starting_date=datetime (year=2024,
                  month=1,
                  day=1,
                  hour=0,
                  minute=0,
                  second=0),
                  select_import_mode="Convert Model",
                  result_well_events=find_object (name="Конструкция скважин",
                  type="gt_wells_events_data"),
                  additional_options=True,
                  apply_welspecs_later=False)
            end_wf_item (index = 29)


        begin_wf_item (index = 30)
        comment_text ("""
необходимо корректно определить директорию внтутри скрипта directory
(если нет папки current, оно появится & актуализируется после запуска SmartCase.exe)
""")
        end_wf_item (index = 30)


        begin_wf_item (index = 31, name = "ГСС")
        schedule_rule_add_apply_script (schedule_strategy=find_object (name="Case",
              type="gt_schedule_rules_data"),
              date_time=datetime (year=2024,
              month=1,
              day=1,
              hour=0,
              minute=0,
              second=0),
              use_rule_name=True,
              rule_name="ГСС в APPLYSCRIPT",
              file_name="start",
              function_name="get_command",
              variables_table=[{"variable" : "r", "value" : "1"}],
              script_text="	import datetime\n	directory=r\"D:\\Orher\\oil\"\n	file_path =f\"{directory}\\\\current\\\\schedule.SCH\"\n	date = get_current_date ( )\n	commands = []\n	write_commands = False\n	next_line_is_date = False\n	date_line = None\n	with open(file_path, \'r\') as file:\n		lines = file.readlines()\n	for line in lines:\n		data = line.split()\n		if next_line_is_date:\n			data_day = data[0]\n			data_month = data[1].strip(\"\'\")\n			data_year = data[2]\n			current_date = datetime.datetime(int(data_year), datetime.datetime.strptime(data_month, \"%b\").month, int(data_day))\n			if current_date == date:\n				write_commands = True\n				date_line = line.strip()\n			else:\n				write_commands = False\n			next_line_is_date = False\n		if data and data[0] == \'DATES\':\n			next_line_is_date = True\n		if write_commands:\n			commands.append(line)\n	if date_line:\n		commands.insert(0, f\"DATES\\n\")\n	if commands:\n		commands.pop()\n	commands_str = \'\'.join(commands)\n	add_keyword (commands_str)\n	")
        end_wf_item (index = 31)



    end_wf_item (index = 27)


    begin_wf_item (index = 33, name = "Управление скважинами")
    schedule_rule_add_apply_script (schedule_strategy=find_object (name="Case",
          type="gt_schedule_rules_data"),
          date_time=datetime (year=2024,
          month=1,
          day=1,
          hour=0,
          minute=0,
          second=0),
          use_rule_name=True,
          rule_name="Управление нефтяными скважинами",
          file_name="control",
          function_name="control_wells",
          variables_table=[{"variable" : "depression_oil", "value" : resolve_variables_in_string (string_with_variables="@DEP_OIL@",
          variables=variables)}, {"variable" : "limit_oil_liq_well", "value" : "500"}, {"variable" : "depression_gas", "value" : resolve_variables_in_string (string_with_variables="@DEP_GAS@",
          variables=variables)}, {"variable" : "limit_gas_gas_well", "value" : "500000"}],
          script_text="	for group in get_all_groups ( ):\n		if group.name == \'OIL\':\n			for w in group.all_wells:\n				# Получаем все скважины на нефть\n				if w.is_stopped ( ):\n					continue\n				BHP_ = wbp9[w] - depression_oil\n				print(f\'Настройка скважины {w.name} Зайбоное давление - {BHP_}\')\n				if BHP_ < 40:\n					print(f\'Отключаем скважину {w.name} Зайбоное давление - {BHP_} ниже 40 бар\')\n					set_well_status (w.name, status = \'stop\')\n					continue\n				\n				add_keyword(\n					\"\"\"\n					WCONPROD\n					\"\"\"+w.name+\"\"\" OPEN LRAT 3* \"\"\"+ str(limit_oil_liq_well) +\"\"\" * \"\"\"+ str(BHP_) +\"\"\" /\n					/\n					\"\"\"\n					)\n				print(f\'Задаем ограничения на скважину {w.name}\')\n				add_keyword(\n					\"\"\"\n					WECON\n					\"\"\"+w.name+\"\"\" 10 * 0.98 8000 * CON+ /\n					/\n					\"\"\"\n					)\n		elif group.name == \'GAS\':\n			for w in group.all_wells:\n				# Получаем все скважины на нефть\n				if w.is_stopped ( ):\n					continue\n				BHP_ = wbp9[w] - depression_gas\n				print(f\'Настройка скважины {w.name} Зайбоное давление - {BHP_}\')\n				if BHP_ < 40:\n					print(f\'Отключаем скважину {w.name} Зайбоное давление - {BHP_} ниже 40 бар\')\n					set_well_status (w.name, status = \'stop\')\n					continue\n				\n				add_keyword(\n					\"\"\"\n					WCONPROD\n					\"\"\"+w.name+\"\"\" OPEN GRAT 2* \"\"\"+ str(limit_gas_gas_well) +\"\"\" 2* \"\"\"+ str(BHP_) +\"\"\" /\n					/\n					\"\"\"\n					)\n				print(f\'Задаем ограничения на скважину {w.name}\')\n				add_keyword(\n					\"\"\"\n					WECON\n					\"\"\"+w.name+\"\"\" * 10000 2* 0.005 CON+ /\n					/\n					\"\"\"\n					)\n		\n")
    end_wf_item (index = 33)


    begin_wf_item (index = 34)
    open_or_reload_dynamic_model (use_model=False,
          model=find_object (name="DynamicModel",
          type="Model_ex"),
          result_name="result")
    end_wf_item (index = 34)


