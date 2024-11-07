from input.variable import Variable
from config.importation import *


class Direct(Variable):
    def __init__(self, variable_file):
        Variable.__init__(self, variable_file)
        self.dict_path_folders = {}

    def create_project_structure(self):
        """
        Создает структуру проекта, включая папки current и history, если они не существуют.
        """
        project_path = self.project_directory
        current_path = os.path.join(project_path, 'current')
        history_path = os.path.join(project_path, 'history')
        os.makedirs(current_path, exist_ok=True)
        os.makedirs(history_path, exist_ok=True)
        trac_graph_path = os.path.join(current_path, 'trac_graph')
        os.makedirs(trac_graph_path, exist_ok=True)
        self.dict_path_folders['current'] = current_path
        self.dict_path_folders['history'] = history_path
        self.dict_path_folders['trac_graph'] = trac_graph_path

    def move_current_to_history(self):
        """
        Переносит содержимое папки current в новую папку history_result_индекс в папке history.
        Индекс определяется как количество файлов в папке history.
        """
        current_path = self.dict_path_folders['current']
        history_path = self.dict_path_folders['history']
        history_files = os.listdir(history_path)
        index = len(history_files)
        history_result_path = os.path.join(history_path, f'history_result_{index}')
        os.makedirs(history_result_path)
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            shutil.move(item_path, history_result_path)

    def is_current_folder_not_empty(self):
        """
        Проверяет, пуста ли папка current.
        Возвращает True, если папка не пуста, и False в противном случае.
        """
        current_path = self.dict_path_folders['current']
        return len(os.listdir(current_path)) > 0

    def clear_current_folder(self):
        """
        Очищает содержимое папки current после переноса.
        """
        current_path = self.dict_path_folders['current']
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        trac_graph_path = os.path.join(current_path, 'trac_graph')
        os.makedirs(trac_graph_path, exist_ok=True)

    def create_json_file(self):
        """
        Создает JSON файл на основе атрибутов, определенных в классе Variable.
        """
        json_path = os.path.join(self.dict_path_folders['current'], 'attributes.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.attributes, json_file, ensure_ascii=False, indent=4)

    def __call__(self, *args, **kwargs):
        """
        Вызывает методы для создания структуры проекта, переноса содержимого папки current в history,
        очистки папки current и переноса лог-файла в папку current.
        """
        Variable.__call__(self)
        self.create_project_structure()
        if self.is_current_folder_not_empty():
            self.move_current_to_history()
            self.clear_current_folder()
        self.create_json_file()