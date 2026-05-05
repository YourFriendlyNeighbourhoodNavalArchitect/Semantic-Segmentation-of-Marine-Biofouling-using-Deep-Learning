from Dataset.SubsetSplit import SubsetSplit
from Various.configurationFile import (
    ALL_PATH,
    METADATA_PATH,
    TESTING_PATH,
    TRAINING_PATH,
    VALIDATION_PATH,
)

SubsetSplit(METADATA_PATH, ALL_PATH, TRAINING_PATH, VALIDATION_PATH, TESTING_PATH)
