from matplotlib.pyplot import subplots
from matplotlib.ticker import FuncFormatter
from math import log10, floor
from Various.configurationFile import MODEL_PATH, CLASS_DICTIONARY

# Ordered class names for per-class reporting.
CLASS_NAMES = list(CLASS_DICTIONARY.keys())


def logResults(epoch, currentLR, trainingMetrics, validationMetrics):
    # CMD outputs of key metrics to sanity-check training.
    print(f'\n{"=" * 50}')
    print(f'  Epoch {epoch}  |  Learning Rate = {currentLR:.6f}')
    print(f'{"=" * 50}')

    for label, metrics in [('Training', trainingMetrics), ('Validation', validationMetrics)]:
        print(f'\n  {label} Metrics:')
        for key, value in metrics.items():
            if isinstance(value, list):
                # Per-class metric — format as table.
                formatted = '  |  '.join(f'{CLASS_NAMES[i]}: {v:.4f}' for i, v in enumerate(value))
                print(f'    {key}: {formatted}')
            elif isinstance(value, float):
                print(f'    {key}: {value:.4f}')
    print()


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
    path = MODEL_PATH / f'trainingPlot{trialNumber}.png'
    figure.savefig(path, dpi = 600, bbox_inches = 'tight')
    print(f'Training plot saved in {path}.')
    return path


def plotMetrics(trainingLossPlot, validationLossPlot, diceScorePlot, IoUScorePlot, trialNumber):
    # Function to create and save plots after training ends.
    figure, axes = subplots(2, 2, figsize = (14, 12))
    figure.subplots_adjust(left = 0.075, right = 0.975, top = 0.925, bottom = 0.075, wspace = 0.2, hspace = 0.3)
    axisOne, axisTwo, axisThree, axisFour = axes.flatten()
    epochs = range(1, len(trainingLossPlot) + 1)

    axisOne.plot(epochs, trainingLossPlot, color = 'blue', label = 'Training Loss')
    axisTwo.plot(epochs, validationLossPlot, color = 'green', label = 'Validation Loss')
    axisThree.plot(epochs, diceScorePlot, color = 'red', label = 'Validation Dice')
    axisFour.plot(epochs, IoUScorePlot, color = 'magenta', label = 'Validation IoU')

    # Configure aesthetics.
    for axis, title, ylabel in zip(
        [axisOne, axisTwo, axisThree, axisFour],
        ['Training Loss', 'Validation Loss', 'Validation Dice Coefficient', 'Validation IoU'],
        ['Loss', 'Loss', 'Dice Coefficient', 'IoU Score']
    ):
        axis.set_xlabel('Epoch', fontsize = 14)
        axis.set_ylabel(ylabel, fontsize = 14)
        axis.set_title(title, fontsize = 16, fontweight = 'bold')
        axis.legend(fontsize = 12)
        axis.grid(True, alpha = 0.3)
        axis.yaxis.set_major_formatter(FuncFormatter(ticksFormat))
        axis.set_xlim(1, len(trainingLossPlot))
        tickPositions = [1] + list(range(25, len(trainingLossPlot) + 1, 25))
        tickLabels = [str(x) for x in tickPositions]
        axis.set_xticks(tickPositions)
        axis.set_xticklabels(tickLabels, fontsize = 12)
        axis.yaxis.set_tick_params(labelsize = 12)

    PNGPath = saveTrainingPlot(figure, trialNumber)
    return PNGPath