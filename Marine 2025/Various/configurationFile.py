from os import getenv
# from dotenv import load_dotenv
from pathlib import Path

# Fetch Labelbox API key.
# load_dotenv()
# LABELBOX_API_KEY = getenv('LABELBOX_API_KEY')

# # Model performs classification amongst the merged classes shown below.

# Original 5-class scheme (before merge) — kept for reference.
CLASS_DICTIONARY = {'Clean Hull': {'index': 0, 'color': [0, 255, 0]},
                    'Slime/Algae': {'index': 1, 'color': [255, 255, 106]},
                    'Calcareous Deposits': {'index': 2, 'color': [255, 87, 51]},
                    'Barnacles/Molluscs': {'index': 3, 'color': [157, 41, 177]},
                    'Background/Other': {'index': 4, 'color': [43, 138, 255]}}

CLASS_DICTIONARY_v2 = {'Clean Hull': {'index': 0, 'color': [0, 255, 0]},
                    'Soft Fouling': {'index': 1, 'color': [255, 255, 106]},
                    'Hard Fouling': {'index': 2, 'color': [255, 87, 51]},
                    'Background/Other': {'index': 3, 'color': [157, 41, 177]}}


# Project configuration variables.
SEED = 42
RESOLUTION = (512, 512)
NUM_CLASSES = len(CLASS_DICTIONARY)
NUM_CLASSES_v2 = len(CLASS_DICTIONARY_v2)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
BATCH_SIZE = 8
WARMUP = 10
PATIENCE = 50
LOG_INTERVAL = 5
# Learning rate from the best Attention U-Net trial (Optuna result).
LEARNING_RATE = 0.00041

# Paths for the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_PATH = PROJECT_ROOT / 'INPUTS' 
ALL_PATH.mkdir(exist_ok=True, parents=True)
METADATA_PATH = ALL_PATH / 'Masks' / 'Metadata.json'
TRAINING_PATH = PROJECT_ROOT / 'INPUTS'  / 'TRAINING'
TRAINING_PATH.mkdir(exist_ok=True, parents=True)
VALIDATION_PATH = PROJECT_ROOT / 'INPUTS'  / 'VALIDATION'
VALIDATION_PATH.mkdir(exist_ok=True, parents=True)
TESTING_PATH = PROJECT_ROOT / 'INPUTS'  / 'TESTING'
TESTING_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH = PROJECT_ROOT / 'OUTPUTS' / 'Trained models'
MODEL_PATH.mkdir(exist_ok=True, parents=True)
VISUALIZATIONS_PATH = PROJECT_ROOT / 'OUTPUTS' / 'Visualizations'
VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)