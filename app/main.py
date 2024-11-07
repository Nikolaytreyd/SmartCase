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
    parser = argparse.ArgumentParser(description="Run the case with a specified variable file.")
    parser.add_argument('variable_file', type=str, help='Path to the JSON file with variables')
    args = parser.parse_args()
    case = Run(variable_file=args.variable_file)
    case()


if __name__ == "__main__":
    main()
