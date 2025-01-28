CLASS_DICTIONARY = {
    # Model performs classification amongst the classes shown below.
    'Clean Hull': {'index': 0, 'color': [0, 255, 0]},
    'Slime': {'index': 1, 'color': [255, 255, 51]},
    'Algae': {'index': 2, 'color': [224, 0, 64]},
    'Calcareous Deposits': {'index': 3, 'color': [224, 128, 0]},
    'Barnacles': {'index': 4, 'color': [96, 0, 64]},
    'Molluscs': {'index': 5, 'color': [0, 224, 192]},
    'Other': {'index': 6, 'color': [255, 255, 255]},
    'Background': {'index': 7, 'color': [0, 39, 255]}
}

# Project configuration variables.
SEED = 29
RESOLUTION = (304, 304)
NUM_CLASSES = len(CLASS_DICTIONARY)
SPLIT_RATIOS = (0.8, 0.2)
ALL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\ALL'
TRAINING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TRAINING'
VALIDATION_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\VALIDATION'
TESTING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TESTING'
MODEL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained models'
VISUALIZATIONS_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Visualizations'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTNlZm42MHIwYzBzMDd4d2dyNmVmYTQ1Iiwib3JnYW5pemF0aW9uSWQiOiJjbTNlZm42MGwwYzByMDd4dzRyZG1jNmZhIiwiYXBpS2V5SWQiOiJjbTR2Z3YybWMwMjdlMDd6ZmV6bTlldjk5Iiwic2VjcmV0IjoiMzZkN2VjZjg1NTgzYjk3ZTBjNjY3MzE5NTg1YjE1MWQiLCJpYXQiOjE3MzQ2MjE1MTQsImV4cCI6MjM2NTc3MzUxNH0.dG52043Nkq9IuuUIKLBWMU7wCRGQF6qtf-X1sGu97yw'