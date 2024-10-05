import os
import re

def renameFiles(maskPath):
    # Simple function that renames .npy files exported from Label Studio.
    # Creates a more coherent structure for the dataset.
    phraseToValue = {
        'No fouling': 1,
        'Light fouling': 2,
        'Heavy fouling': 3,
        'Sea': 4
    }

    for oldFilename in os.listdir(maskPath):
        # Standard Label Studio output.
        match = re.match(r'task-(\d+)-annotation-\d+-by-\d+-tag-', oldFilename)
        if match:
            X = match.group(1)
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
            print(f'Renamed {oldFilename} to {newFilename}.')
        else:
            print(f'Skipped {oldFilename} (does not match expected pattern).')

maskPath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\Masks'
renameFiles(maskPath)
