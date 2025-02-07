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
SPLIT_RATIOS = [0.75, 0.25]
BATCH_SIZE = 16
WARMUP = 50
PATIENCE = 100
ALL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\ALL'
METADATA_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\ALL\Masks\Metadata.json'
TRAINING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\STRATIFIED\TRAINING'
VALIDATION_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\STRATIFIED\VALIDATION'
TESTING_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TESTING'
MODEL_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained models'
VISUALIZATIONS_PATH = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Visualizations'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTNlZm42MHIwYzBzMDd4d2dyNmVmYTQ1Iiwib3JnYW5pemF0aW9uSWQiOiJjbTNlZm42MGwwYzByMDd4dzRyZG1jNmZhIiwiYXBpS2V5SWQiOiJjbTR2Z3YybWMwMjdlMDd6ZmV6bTlldjk5Iiwic2VjcmV0IjoiMzZkN2VjZjg1NTgzYjk3ZTBjNjY3MzE5NTg1YjE1MWQiLCJpYXQiOjE3MzQ2MjE1MTQsImV4cCI6MjM2NTc3MzUxNH0.dG52043Nkq9IuuUIKLBWMU7wCRGQF6qtf-X1sGu97yw'