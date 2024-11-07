from config.importation import *

def global_random(number: int = 0, option: bool = True):
    if option:
        np.random.seed(number)

def warnings_off():
    fw('ignore')
