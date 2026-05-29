from Dataset.SubsetSplit import SubsetSplit
from Dataset.remapClasses import remapAllSubsets
from Various.configurationFile import (
    ALL_PATH,
    METADATA_PATH,
    TESTING_PATH,
    TRAINING_PATH,
    VALIDATION_PATH,
)

SubsetSplit(METADATA_PATH, ALL_PATH, TRAINING_PATH, VALIDATION_PATH, TESTING_PATH)
# Step 2: Remap masks from 5-class to 4-class (merged) scheme.
# Slime/Algae → Soft Fouling, Calcareous Deposits + Barnacles/Molluscs → Hard Fouling.
remapAllSubsets(TRAINING_PATH, VALIDATION_PATH, TESTING_PATH)
 