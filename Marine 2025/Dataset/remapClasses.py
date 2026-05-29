import numpy as np
from pathlib import Path
from Various.configurationFile import CLASS_DICTIONARY

# Mapping from the original 5-class scheme to the merged 4-class scheme.
# Original:  Clean Hull (0), Slime/Algae (1), Calcareous Deposits (2), Barnacles/Molluscs (3), Background/Other (4)
# Merged:    Clean Hull (0), Soft Fouling (1), Hard Fouling (2), Background/Other (3)

REMAP = np.array([0, 1, 2, 2, 3], dtype=np.uint8)

# Merged class names for reporting.
MERGED_CLASS_NAMES = list(CLASS_DICTIONARY.keys())


def remapMasks(maskDirectory: Path):
    """Remap all .npy masks in a directory from 5-class to 4-class scheme in-place."""
    maskPaths = sorted(maskDirectory.glob('*.npy'))
    if not maskPaths:
        print(f'  No masks found in {maskDirectory}.')
        return {}, 0, 0

    remappedCount = 0
    totalPixels = 0
    # Track two things: pixel counts per class, and image occurrence per class.
    pixelCounts = {i: 0 for i in range(len(MERGED_CLASS_NAMES))}
    imageCounts = {i: 0 for i in range(len(MERGED_CLASS_NAMES))}

    for maskPath in maskPaths:
        mask = np.load(maskPath)
        uniqueValues = np.unique(mask)
        if uniqueValues.max() > 4:
            print(f'  WARNING: {maskPath.name} contains unexpected class index {uniqueValues.max()}. Skipping.')
            continue

        remappedMask = REMAP[mask]
        np.save(maskPath, remappedMask)
        remappedCount += 1
        totalPixels += remappedMask.size

        # Pixel-level counts.
        for classIndex in range(len(MERGED_CLASS_NAMES)):
            classPixels = int(np.sum(remappedMask == classIndex))
            pixelCounts[classIndex] += classPixels
            if classPixels > 0:
                imageCounts[classIndex] += 1

    return pixelCounts, imageCounts, remappedCount, totalPixels


def _printDistribution(subsetName, pixelCounts, imageCounts, numImages, totalPixels):
    """Print pixel-level and image-level class distribution for a subset."""
    print(f'\n  {subsetName} ({numImages} images, {totalPixels:,} total pixels):')
    print(f'  {"Class":<20} {"Pixels":>14} {"Pixel %":>9} {"Images":>8} {"Image %":>9}')
    print(f'  {"-" * 62}')
    for classIndex, className in enumerate(MERGED_CLASS_NAMES):
        px = pixelCounts[classIndex]
        pxPct = (px / totalPixels * 100) if totalPixels > 0 else 0
        im = imageCounts[classIndex]
        imPct = (im / numImages * 100) if numImages > 0 else 0
        print(f'  {className:<20} {px:>14,} {pxPct:>8.2f}% {im:>8} {imPct:>8.1f}%')


def remapAllSubsets(*subsetPaths: Path):
    """Remap masks across all dataset subsets and print class distributions."""
    print('\n' + '=' * 60)
    print('  Class Remapping: 5-class → 4-class (merged)')
    print('=' * 60)

    # Accumulators for the global summary.
    globalPixelCounts = {i: 0 for i in range(len(MERGED_CLASS_NAMES))}
    globalImageCounts = {i: 0 for i in range(len(MERGED_CLASS_NAMES))}
    globalImages = 0
    globalPixels = 0

    for subsetPath in subsetPaths:
        maskDirectory = subsetPath / 'Masks'
        subsetName = subsetPath.name
        if not maskDirectory.exists():
            print(f'  Mask directory not found: {maskDirectory}')
            continue

        pixelCounts, imageCounts, numRemapped, totalPixels = remapMasks(maskDirectory)
        if numRemapped == 0:
            continue

        _printDistribution(subsetName, pixelCounts, imageCounts, numRemapped, totalPixels)

        # Accumulate.
        for i in range(len(MERGED_CLASS_NAMES)):
            globalPixelCounts[i] += pixelCounts[i]
            globalImageCounts[i] += imageCounts[i]
        globalImages += numRemapped
        globalPixels += totalPixels

    # Global summary.
    if globalImages > 0:
        _printDistribution('TOTAL', globalPixelCounts, globalImageCounts, globalImages, globalPixels)
    print()