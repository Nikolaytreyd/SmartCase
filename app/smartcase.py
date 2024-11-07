from modules.wellcover import Cover
from config.config import warnings_off
from config.importation import *

class Run (Cover):
    def __init__(self, variable_file):
        warnings_off()
        Cover.__init__(self, variable_file)

    def __call__(self, *args, **kwargs):
        Cover.__call__(self)

def main():
    case = Run(variable_file=r"D:\Orher\oil\variable.json")
    case()


if __name__ == "__main__":
    main()
