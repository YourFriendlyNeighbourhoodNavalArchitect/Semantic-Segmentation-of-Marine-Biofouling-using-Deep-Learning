import matplotlib.pyplot as plt
from numpy import array, polyfit, poly1d, linspace, arange
from json import load
from configurationFile import MODEL_PATH

def plotExperimentResults(rootPath, studyNumber):
    # Plot the final validation loss against the initial learning rate, for all trials in each experiment.
    resultsFilename = 'trialLog.json'
    studyPath = f'STUDY ({studyNumber})'
    JSONPath = rootPath / studyPath / resultsFilename
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
    x = array(initialLearningRates)
    y = array(validationLosses)
    coeffs = polyfit(x, y, 1)
    trendline = poly1d(coeffs)
    xFit = linspace(min(x), max(x), 100)
    yFit = trendline(xFit)
    
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, label = 'Trials', color = 'blue', alpha = 0.7)
    plt.plot(xFit, yFit, linestyle = '--', color = 'red', label = 'Trendline')
    
    plt.xlabel('Initial Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title(f'Experiment ({studyNumber}) Results Visualisation')
    plt.xticks(linspace(5e-5, 5e-4, num = len(initialLearningRates))[::3])
    plt.yticks(arange(min(y), max(y) + (max(y) - min(y)) / 10, (max(y) - min(y)) / 10))
    
    plt.legend()
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.tight_layout()
    outputPath = JSONPath.parent / 'experimentResults.png'
    plt.savefig(outputPath, dpi = 600)
    plt.close()

for i in range(1, 5):
    plotExperimentResults(MODEL_PATH, i)
