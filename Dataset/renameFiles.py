import os
from re import match

def renameFiles(maskPath):
    # Simple function that renames .npy files exported from Label Studio.
    # Creates a more coherent structure for the dataset.
    phraseToValue = {
        'No fouling': 1,
        'Light fouling': 2,
        'Heavy fouling': 3,
        'Background': 4
    }

    for oldFilename in os.listdir(maskPath):
        # Standard Label Studio output.
        search = match(r'task-(\d+)-annotation-\d+-by-\d+-tag-', oldFilename)
        if search:
            X = search.group(1)
            Y = 0
            for phrase, value in phraseToValue.items():
                if phrase in oldFilename:
                    Y = value
                    break

            newFilename = f'{X}.{Y}.npy'
            oldPath = os.path.join(maskPath, oldFilename)
            newPath = os.path.join(maskPath, newFilename)
            os.rename(oldPath, newPath)
            # CMD output to mark success or failure.
            print(f"Renamed {oldFilename} to {newFilename}.")
        else:
            print(f"Skipped {oldFilename} (does not match expected pattern).")

maskPath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TRAINING\Masks'
renameFiles(maskPath)
