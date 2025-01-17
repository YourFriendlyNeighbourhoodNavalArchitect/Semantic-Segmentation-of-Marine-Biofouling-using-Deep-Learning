CLASS_DICTIONARY = {
    # Model performs classification amongst the classes shown below.
    'Clean Hull': {'index': 0, 'colour': [0, 255, 0]},
    'Slime': {'index': 1, 'colour': [255, 255, 51]},
    'Algae': {'index': 2, 'colour': [224, 0, 64]},
    'Calcareous Deposits': {'index': 3, 'colour': [224, 128, 0]},
    'Barnacles': {'index': 4, 'colour': [96, 0, 64]},
    'Molluscs': {'index': 5, 'colour': [0, 224, 192]},
    'Other': {'index': 6, 'colour': [255, 255, 255]},
    'Background': {'index': 7, 'colour': [0, 39, 255]}
}

# Project configuration variables.
SEED = 29
RESOLUTION = (304, 304)
NUM_CLASSES = len(CLASS_DICTIONARY)
SPLIT_RATIOS = (0.7, 0.15, 0.15)
ALL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\ALL'
TRAINING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TRAINING'
VALIDATION_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\VALIDATION'
TESTING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TESTING'
MODEL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained models'
VISUALISATIONS_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Visualizations'
API_KEY = 'REMOVED_LABELBOX_API_KEY'