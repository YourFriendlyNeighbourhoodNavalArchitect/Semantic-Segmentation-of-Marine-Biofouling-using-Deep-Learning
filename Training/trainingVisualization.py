import matplotlib.pyplot as plt

def logResults(epoch, trainingLoss, validationLoss, diceScore, IoUScore):
    # CMD outputs of key metrics to sanity-check training.
    print(f"\nEpoch {epoch + 1} results:\n"
        f"Training loss: {trainingLoss:.4f}\n"
        f"Validation loss: {validationLoss:.4f}\n"
        f"Dice coefficient: {diceScore:.4f}\n"
        f"IoU score: {IoUScore:.4f}\n")

def plotMetrics(trainingLossPlot, validationLossPlot, diceScorePlot, IoUScorePlot, epoch):
    # Initialize the figure and axes if it's the first call.
    # Otherwise reset the axes for subsequent trial.
    if epoch == 0:
        plt.ion()
        if not hasattr(plotMetrics, 'figure'):
            plotMetrics.figure, axes = plt.subplots(2, 2, figsize = (12, 10))
            plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour = axes.flatten()
            titles = ["Training Loss", "Validation Loss", "Dice Coefficient", "IoU Score"]
            for axis, title in zip([plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour], titles):
                axis.set_xlabel("Epoch")
                axis.set_title(title)
                axis.grid(True)
        else:
            titles = ["Training Loss", "Validation Loss", "Dice Coefficient", "IoU Score"]
            for axis, title in zip([plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour], titles):
                axis.cla()
                axis.set_xlabel("Epoch")
                axis.set_title(title)
                axis.grid(True)
    
    epochs = range(1, epoch + 2)
    plotMetrics.axisOne.plot(epochs, trainingLossPlot, color = 'blue', label = 'Training Loss')
    plotMetrics.axisTwo.plot(epochs, validationLossPlot, color = 'green', label = 'Validation Loss')
    plotMetrics.axisThree.plot(epochs, diceScorePlot, color = 'red', label = 'Dice Coefficient')
    plotMetrics.axisFour.plot(epochs, IoUScorePlot, color = 'magenta', label = 'IoU Score')

    for axis in [plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour]:
        axis.set_xlim(1, epoch + 2)
        axis.set_xticks(epochs)

    plotMetrics.figure.canvas.draw()
    plotMetrics.figure.canvas.flush_events()
