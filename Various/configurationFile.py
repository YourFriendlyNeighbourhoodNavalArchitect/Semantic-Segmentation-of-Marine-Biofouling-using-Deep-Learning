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
NUM_CLASSES = len(CLASS_DICTIONARY)
ALL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\ALL'
TRAINING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TRAINING'
TESTING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TESTING\Images'
MODEL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained model'
VISUALISATIONS_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Visualizations'
API_KEY = 'REMOVED_LABELBOX_API_KEY'