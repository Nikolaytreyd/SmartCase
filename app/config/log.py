from config.importation import *


class Logger:
    def __init__(self, enable_file_logging=True):
        self.temp_dir = tempfile.mkdtemp()
        self.log_directory = os.path.join(self.temp_dir, 'logs')
        self.enable_file_logging = enable_file_logging
        self.call_counter = {}
        self.log_buffer = StringIO()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s | Call #%(call_number)d | %(message)s',
            datefmt='%d:%m:%Y %H:%M:%S')
        buffer_handler = logging.StreamHandler(self.log_buffer)
        buffer_handler.setFormatter(formatter)
        logger.addHandler(buffer_handler)
        if self.enable_file_logging:
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory)
            log_file = os.path.join(self.log_directory, 'log.log')
            if os.path.exists(log_file):
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.truncate(0)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def log_info(self, message, func_name=None):
        if func_name is None:
            func_name = self.log_info.__name__
        if func_name not in self.call_counter:
            self.call_counter[func_name] = 0
        self.call_counter[func_name] += 1
        call_number = self.call_counter[func_name]
        extra = {
            'call_number': call_number
        }
        self.logger.info(f"{message}", extra=extra)

    def log_warning(self, message, func_name=None):
        if func_name is None:
            func_name = self.log_warning.__name__
        if func_name not in self.call_counter:
            self.call_counter[func_name] = 0
        self.call_counter[func_name] += 1
        call_number = self.call_counter[func_name]
        extra = {
            'call_number': call_number
        }

        self.logger.warning(f"{message}", extra=extra)

    def log_error(self, message, func_name=None):
        if func_name is None:
            func_name = self.log_error.__name__
        if func_name not in self.call_counter:
            self.call_counter[func_name] = 0
        self.call_counter[func_name] += 1
        call_number = self.call_counter[func_name]
        extra = {
            'call_number': call_number
        }
        self.logger.error(f"{message}", extra=extra)

    def save_log_to_file(self, path):
        """
        Сохраняет содержимое буфера логов в файл в указанном пути.
        :param path: Путь к папке current.
        """
        log_file = os.path.join(path, 'log.log')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(self.log_buffer.getvalue())

    def cleanup(self):
        shutil.rmtree(self.temp_dir)