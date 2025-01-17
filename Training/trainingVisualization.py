from matplotlib.pyplot import subplots
from matplotlib.ticker import FuncFormatter, MultipleLocator
from math import log10, floor
from os.path import join
from configurationFile import VISUALISATIONS_PATH

def logResults(epoch, trainingMetrics, validationMetrics):
    # CMD outputs of key metrics to sanity-check training.
    print(f'\nEpoch {epoch + 1} results:')
    print('Training Metrics:')
    for key, value in trainingMetrics.items():
        print(f'{key}: {value:.4f}')
    print('Validation Metrics:')
    for key, value in validationMetrics.items():
        print(f'{key}: {value:.4f}')
    print('\n')

def ticksFormat(x, pos):
    # Dummy function to format y-axis ticks.
    if abs(x) >= 1000:
        exponent = floor(log10(abs(x)))
        mantissa = x / (10**exponent)
        mantissa = f'{mantissa:.1f}'.rstrip('0').rstrip('.')
        return r'${} \cdot 10^{{{}}}$'.format(mantissa, exponent)
    else:
        return f'{x:.3f}'.rstrip('0').rstrip('.')

def saveTrainingPlot(figure, trialNumber):
    # Save the final plot of a trial as a PNG.
    path = join(VISUALISATIONS_PATH, f'trainingPlot{trialNumber}.png')
    figure.savefig(path, dpi = 600)
    print(f'Training plot saved in {path}.')
    return path

def plotMetrics(trainingLossPlot, validationLossPlot, diceScorePlot, IoUScorePlot, trialNumber):
    # Function to create and save plots after training ends.
    figure, axes = subplots(2, 2, figsize = (12, 12))
    figure.subplots_adjust(left = 0.075, right = 0.975, top = 0.925, bottom = 0.075, wspace = 0.15, hspace = 0.25)
    axisOne, axisTwo, axisThree, axisFour = axes.flatten()
    epochs = range(1, len(trainingLossPlot) + 1)

    # Plot metrics.
    axisOne.plot(epochs, trainingLossPlot, color = 'blue', label = 'Training Loss')
    axisTwo.plot(epochs, validationLossPlot, color = 'green', label = 'Validation Loss')
    axisThree.plot(epochs, diceScorePlot, color = 'red', label = 'Dice Coefficient')
    axisFour.plot(epochs, IoUScorePlot, color = 'magenta', label = 'IoU Score')

    # Configure aesthetics.
    for axis, title in zip([axisOne, axisTwo, axisThree, axisFour], 
                           ['Training Loss', 'Validation Loss', 'Dice Coefficient', 'IoU Score']):
        axis.set_xlabel('Epoch', fontsize = 14)
        axis.set_title(title, fontsize = 14, fontweight = 'bold')
        axis.grid(True)
        axis.yaxis.set_major_formatter(FuncFormatter(ticksFormat))
        axis.set_xlim(1, len(trainingLossPlot))
        axis.xaxis.set_major_locator(MultipleLocator(1))
        tickPositions = list(range(1, len(trainingLossPlot) + 1))
        tickLabels = [str(x) if x % 5 == 0 or x == 1 else '' for x in tickPositions]
        axis.set_xticks(tickPositions)
        axis.set_xticklabels(tickLabels, fontsize = 14)
        axis.yaxis.set_tick_params(labelsize = 14) 

    # Save figure.
    PNGPath = saveTrainingPlot(figure, trialNumber)
    return PNGPath