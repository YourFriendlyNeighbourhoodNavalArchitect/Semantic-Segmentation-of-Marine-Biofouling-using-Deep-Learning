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
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTNlZm42MHIwYzBzMDd4d2dyNmVmYTQ1Iiwib3JnYW5pemF0aW9uSWQiOiJjbTNlZm42MGwwYzByMDd4dzRyZG1jNmZhIiwiYXBpS2V5SWQiOiJjbTR2Z3YybWMwMjdlMDd6ZmV6bTlldjk5Iiwic2VjcmV0IjoiMzZkN2VjZjg1NTgzYjk3ZTBjNjY3MzE5NTg1YjE1MWQiLCJpYXQiOjE3MzQ2MjE1MTQsImV4cCI6MjM2NTc3MzUxNH0.dG52043Nkq9IuuUIKLBWMU7wCRGQF6qtf-X1sGu97yw'