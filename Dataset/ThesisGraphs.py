import numpy as np
from json import load
from matplotlib.pyplot import subplots, savefig, close
from seaborn import heatmap
from os.path import join
from textwrap import TextWrapper
from configurationFile import CLASS_DICTIONARY, NUM_CLASSES, METADATA_PATH, VISUALIZATIONS_PATH

class ThesisGraphs:
    def __init__(self, metadataPath, outputDirectory):
        self.metadataPath = metadataPath
        self.outputDirectory = outputDirectory
        self.metadata = self.loadMetadata()
        self.totalImages = len(self.metadata)
        self.classNames = {value['index']: key for key, value in CLASS_DICTIONARY.items()}
        self.wrapper = TextWrapper(width = 10, break_long_words = False, break_on_hyphens = False)
        self.classOccurrences = np.zeros(NUM_CLASSES, dtype = int)
        self.correlationMatrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype = float)
        self.uniqueClassesCount = np.zeros(self.totalImages, dtype = int)
        self.generateGraphs()

    def loadMetadata(self):
        with open(self.metadataPath, 'r') as f:
            return load(f)

    def analyzeMetadata(self):
        # Use broadcasting for computational efficiency.
        for ID, (_, data) in enumerate(self.metadata.items()):
            uniqueClasses = np.array(data['uniqueClassIndices'])
            self.classOccurrences[uniqueClasses] += 1
            self.uniqueClassesCount[ID] = len(uniqueClasses)
            for i in uniqueClasses:
                self.correlationMatrix[i, uniqueClasses] += 1

        # Normalize the correlation matrix row-wise to calculate probabilities.
        rowSums = self.classOccurrences[:, None]
        self.correlationMatrix = np.divide(self.correlationMatrix, rowSums,
                                           out = np.zeros_like(self.correlationMatrix), where = rowSums > 0)

    def plotClassOccurrences(self):
        # Sort classes by occurrence frequency, in descending order.
        sortedOccurrences = sorted(enumerate(self.classOccurrences), key = lambda x: x[1], reverse = True)
        classIndices, occurrenceCounts = zip(*sortedOccurrences)
        classLabels = [self.classNames[index] for index in classIndices]
        classLabels = ['\n'.join(self.wrapper.wrap(label)) for label in classLabels]
        percentages = np.array(occurrenceCounts) / self.totalImages * 100
        # Retrieve class-specific colors.
        colors = [tuple(i / 255 for i in CLASS_DICTIONARY[self.classNames[index]]['color']) for index in classIndices]

        figure, axes = subplots(figsize = (10, 6))
        bars = axes.bar(classLabels, occurrenceCounts, color = colors, edgecolor = 'black')
        axes.set_ylabel('Image Count', fontsize = 14)
        axes.set_title('Class Occurrence Frequency', fontsize = 16, fontweight = 'bold')
        axes.tick_params(axis = 'x', labelsize = 12)
        axes.tick_params(axis = 'y', labelsize = 12)
        axes.legend(handles = [axes.plot([], [])[0]], labels = [f'Dataset Size = {self.totalImages}'],
                    loc = 'upper right', fontsize = 12, handletextpad = 0, handlelength = 0, frameon = False)
        # Add raw counts and percentages as text labels above bars.
        for bar, count, percentage in zip(bars, occurrenceCounts, percentages):
            label = f'{count} ({percentage:.1f}%)'
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, 
                      ha = 'center', va = 'bottom', fontsize = 10, color = 'black')
        figure.tight_layout()
        outputPath = join(self.outputDirectory, 'Class occurence bar chart.png')
        savefig(outputPath, dpi = 600)
        close()

    def plotClassCorrelation(self):
        # Plot heatmap using precomputed normalized correlation matrix.
        classLabels = [self.classNames[index] for index in range(NUM_CLASSES)]
        classLabels = ['\n'.join(self.wrapper.wrap(label)) for label in classLabels]
        figure, axes = subplots(figsize = (14, 12))
        heatmap(self.correlationMatrix, annot = True, fmt = '.2f', cmap = 'coolwarm',
                xticklabels = classLabels, yticklabels = classLabels)
        axes.set_yticklabels(classLabels, rotation = 0, va = 'center')
        axes.set_title('Class Correlation Matrix', fontsize = 16, fontweight = 'bold')
        axes.tick_params(axis = 'x', labelsize = 12)
        axes.tick_params(axis = 'y', labelsize = 12)
        figure.tight_layout()
        outputPath = join(self.outputDirectory, 'Class correlation matrix.png')
        savefig(outputPath, dpi = 600)
        close()

    def plotUniqueClassesPerImage(self):
        # Sort images by count of unique classes per image.
        uniqueCountDistribution = np.bincount(self.uniqueClassesCount)[1:]
        labels = range(1, NUM_CLASSES + 1)
        percentages = uniqueCountDistribution / self.totalImages * 100

        figure, axes = subplots(figsize = (10, 6))
        bars = axes.bar(labels, uniqueCountDistribution, color = 'skyblue', edgecolor = 'black')
        axes.set_ylabel('Image Count', fontsize = 14)
        axes.set_title('Count of Unique Classes per Image', fontsize = 16, fontweight = 'bold')
        axes.tick_params(axis = 'x', labelsize = 12)
        axes.tick_params(axis = 'y', labelsize = 12)
        axes.legend(handles = [axes.plot([], [])[0]], labels = [f'Dataset Size = {self.totalImages}'],
                    loc = 'upper right', fontsize = 12, handletextpad = 0, handlelength = 0, frameon = False)

        for bar, count, percentage in zip(bars, uniqueCountDistribution, percentages):
            label = f'{count} ({percentage:.1f}%)'
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, 
                      ha = 'center', va = 'bottom', fontsize = 10, color = 'black')
        figure.tight_layout()
        outputPath = join(self.outputDirectory, 'Count of unique classes per image.png')
        savefig(outputPath, dpi = 600)
        close()

    def generateGraphs(self):
        self.analyzeMetadata()
        self.plotClassOccurrences()
        self.plotClassCorrelation()
        self.plotUniqueClassesPerImage()

ThesisGraphs(METADATA_PATH, VISUALIZATIONS_PATH)