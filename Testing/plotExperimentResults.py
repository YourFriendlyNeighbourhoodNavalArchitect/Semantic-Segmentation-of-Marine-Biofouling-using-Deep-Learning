import numpy as np
import matplotlib.pyplot as plt
from json import load
from os.path import dirname, join

def plotExperimentResults(JSONPath):
    # Plot the final validation loss against the initial learning rate, for all trials in each experiment.
    with open(JSONPath, 'r') as f:
        data = load(f)
    
    initialLearningRates = []
    validationLosses = []
    for _, epochs in data.items():
        lastEpoch = max(epochs.keys(), key = lambda x: int(x.split()[-1]))
        initialLearningRate = epochs['Epoch 50']['learningRate']
        validationLoss = epochs[lastEpoch]['validationMetrics']['Loss']
        initialLearningRates.append(initialLearningRate)
        validationLosses.append(validationLoss)
    
    # Draw a trendline to observe emerging patterns.
    x = np.array(initialLearningRates)
    y = np.array(validationLosses)
    coeffs = np.polyfit(x, y, 1)
    trendline = np.poly1d(coeffs)
    xFit = np.linspace(min(x), max(x), 100)
    yFit = trendline(xFit)
    
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, label = 'Trials', color = 'blue', alpha = 0.7)
    plt.plot(xFit, yFit, linestyle = '--', color = 'red', label = 'Trendline')
    
    plt.xlabel('Initial Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title('Experiment Results Visualisation')
    plt.xticks(np.linspace(5e-5, 5e-4, num = len(initialLearningRates))[::3])
    plt.yticks(np.arange(min(y), max(y) + (max(y) - min(y)) / 10, (max(y) - min(y)) / 10))
    
    plt.legend()
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.tight_layout()
    outputDirectory = dirname(JSONPath)
    outputPath = join(outputDirectory, 'experimentResults.png')
    plt.savefig(outputPath, dpi = 600)
    plt.close()

JSONPath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained models\STUDY (1)\trialLog.json'
plotExperimentResults(JSONPath)