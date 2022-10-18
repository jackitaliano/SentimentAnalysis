import torch
import numpy as np
import os
from pylab import rcParams
from configparser import ConfigParser

# config
CONFIG = ConfigParser()
try:
    with open('config.ini', 'r') as f:
        CONFIG.read_file(f)
    if CONFIG.getboolean('debug', 'enabled'):
        print("--DEBUG ENABLED")
    if CONFIG.getboolean('analysis', 'enabled'):
        print("--ANALYSIS ENABLED")

except IOError as e:
    print(e)
    print(f'Error: config file not found, exiting...')
    exit()

# debug
DEBUG = CONFIG.getboolean('debug', 'enabled')

# data analysis
ANALYSIS = CONFIG.getboolean('analysis', 'enabled')

# Setup
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

OS = CONFIG.get('device', 'os')
USE_GPU = CONFIG.getboolean('device', 'use_gpu')
if OS == 'mac':
    DEVICE = torch.device(
        "mps" if (USE_GPU and torch.backends.mps.is_availabe()) else "cpu")
else:
    DEVICE = torch.device(
        "cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
if DEBUG:
    print(f'DEVICE: {DEVICE}')

# Files paths
data_folder = CONFIG.get('files', 'folder')
review_file_name = CONFIG.get('files', 'reviews')
apps_file_name = CONFIG.get('files', 'apps')
REVIEWS_F_PATH = os.path.join(data_folder, review_file_name)
APPS_F_PATH = os.path.join(data_folder, apps_file_name)

# Model params
PRE_TRAINED_MODEL_NAME = CONFIG.get('model', 'pretrained_model_name')
MAX_LEN = CONFIG.getint('model', 'max_len')
BATCH_SIZE = CONFIG.getint('model', 'batch_size')
EPOCHS = CONFIG.getint('model', 'epochs')
if DEBUG:
    print(f'Pre-Trained Model name: {PRE_TRAINED_MODEL_NAME}')
    print(f'Max length tokens: {MAX_LEN}')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'Num epochs: {EPOCHS}')
