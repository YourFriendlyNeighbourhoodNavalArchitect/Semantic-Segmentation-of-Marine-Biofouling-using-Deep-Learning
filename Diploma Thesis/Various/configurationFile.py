from os import getenv
from dotenv import load_dotenv
from pathlib import Path

# Fetch Labelbox API key.
load_dotenv()
LABELBOX_API_KEY = getenv('LABELBOX_API_KEY')

# Model performs classification amongst the classes shown below.
CLASS_DICTIONARY = {'Clean Hull': {'index': 0, 'color': [0, 255, 0]},
                    'Slime/Algae': {'index': 1, 'color': [255, 255, 106]},
                    'Calcareous Deposits': {'index': 2, 'color': [255, 87, 51]},
                    'Barnacles/Molluscs': {'index': 3, 'color': [157, 41, 177]},
                    'Background/Other': {'index': 4, 'color': [43, 138, 255]}}

# Project configuration variables.
SEED = 4
RESOLUTION = (304, 304)
NUM_CLASSES = len(CLASS_DICTIONARY)
SPLIT_RATIOS = (0.75, 0.25)
BATCH_SIZE = 16
WARMUP = 50
PATIENCE = 100

# Paths for the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_PATH = PROJECT_ROOT / 'INPUTS' / 'ALL'
METADATA_PATH = ALL_PATH / 'Masks' / 'Metadata.json'
TRAINING_PATH = PROJECT_ROOT / 'INPUTS' / 'STRATIFIED' / 'TRAINING'
VALIDATION_PATH = PROJECT_ROOT / 'INPUTS' / 'STRATIFIED' / 'VALIDATION'
TESTING_PATH = PROJECT_ROOT / 'INPUTS' / 'TESTING'
MODEL_PATH = PROJECT_ROOT / 'OUTPUTS' / 'Trained models'
VISUALIZATIONS_PATH = PROJECT_ROOT / 'OUTPUTS' / 'Visualizations'
