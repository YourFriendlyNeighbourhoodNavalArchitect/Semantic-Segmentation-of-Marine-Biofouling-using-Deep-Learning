import os
import re

def renameFiles(directory):
    # Simple function that renames .npy files exported from Label Studio.
    # Creates a more coherent structure for the dataset.
    phraseToValue = {
        'No fouling': 1,
        'Light fouling': 2,
        'Heavy fouling': 3,
        'Sea': 4
    }

    for filename in os.listdir(directory):
        # Standard Label Studio output.
        match = re.match(r'task-(\d+)-annotation-\d+-by-\d+-tag-', filename)
        if match:
            X = match.group(1)
            Y = 0
            for phrase, value in phraseToValue.items():
                if phrase in filename:
                    Y = value
                    break

            newFilename = f'{X}.{Y}.npy'
            oldPath = os.path.join(directory, filename)
            newPath = os.path.join(directory, newFilename)
            os.rename(oldPath, newPath)
            # CMD output to mark success or failure.
            print(f'Renamed {filename} to {newFilename}.')
        else:
            print(f'Skipped {filename} (does not match expected pattern).')

renameFiles(r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\Dataset\Masks')
