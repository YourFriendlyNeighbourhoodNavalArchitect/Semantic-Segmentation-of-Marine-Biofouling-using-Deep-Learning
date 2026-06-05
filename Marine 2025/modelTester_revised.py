from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from onnxruntime import InferenceSession
from torch import from_numpy
from torchmetrics.functional.classification import multiclass_confusion_matrix

from Dataset.MyDataset import MyDataset
from Training.computeMetrics import computeMetrics
from Training.trainingInitialization import setupDevice
from Various.configurationFile import (
    MODEL_PATH,
    PROJECT_ROOT,
    TESTING_PATH,
    CLASS_DICTIONARY_new,
    NUM_CLASSES_new,
)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 600,
    }
)

CLASS_NAMES = list(CLASS_DICTIONARY_new.keys())
SCALAR_KEYS = [
    "Dice Coefficient",
    "Mean Dice",
    "IoU",
    "Accuracy",
    "Precision",
    "Recall",
]
PER_CLASS_KEYS = [
    "Per Class Dice",
    "Per Class IoU",
    "Per Class Accuracy",
    "Per Class Precision",
    "Per Class Recall",
]


class ModelTester:
    def __init__(
        self,
        modelPath: Path,
        rootPath: Path,
        outputDirectory: Path,
        device: str,
    ) -> None:
        self.modelPath = modelPath
        self.rootPath = rootPath
        self.device = device
        self.outputDirectory = outputDirectory
        self.outputDirectory.mkdir(exist_ok=True, parents=True)
        self.dataset = self.loadDataset()
        self.classColors = CLASS_DICTIONARY_new
        self.session = self.createSession()
        self.runInference()

    def createSession(self):
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            return InferenceSession(self.modelPath, providers=providers)
        except Exception as e:
            print(f"Error while creating ONNX session: {e}")
            raise

    def loadDataset(self):
        try:
            dataset = MyDataset(self.rootPath, augmentationFlag=False)
            print(f"Total number of samples in the dataset: {len(dataset)}")
        except Exception as e:
            print(f"Error while loading test images: {e}")
            raise
        else:
            return dataset

    def calculateClassCoverage(self, mask: np.ndarray) -> dict:
        totalPixels = int(np.prod(mask.shape))
        unique, counts = np.unique(mask, return_counts=True)
        classCoverage = {}
        for className, properties in self.classColors.items():
            classIndex = properties["index"]
            classPixels = (
                counts[np.where(unique == classIndex)][0] if classIndex in unique else 0
            )
            classCoverage[className] = (classPixels / totalPixels) * 100
        return classCoverage

    def classIndicesToRGB(self, mask: np.ndarray):
        height, width = mask.shape
        RGBMask = np.zeros((height, width, 3), dtype=np.uint8)
        for properties in self.classColors.values():
            classIndex = properties["index"]
            color = properties["color"]
            RGBMask[mask == classIndex] = color
        return RGBMask

    def generateLegend(self, coverage: dict) -> tuple[list, list]:
        filteredCoverage = {
            className: pct for className, pct in coverage.items() if pct > 0
        }
        legendLabels = [f"{name}: {pct:.2f}%" for name, pct in filteredCoverage.items()]
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markerfacecolor=np.array(self.classColors[name]["color"]) / 255,
            )
            for name in filteredCoverage
        ]
        return legendLabels, handles

    def plotResults(self, image, prediction, groundTruth, index, perImageDice):
        predictedMask = self.classIndicesToRGB(prediction)
        groundTruthMask = self.classIndicesToRGB(groundTruth)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.04)

        axes[0].imshow(image.permute(1, 2, 0).numpy())
        axes[0].axis("off")

        axes[1].imshow(predictedMask)
        axes[1].axis("off")

        axes[2].imshow(groundTruthMask)
        axes[2].axis("off")

        # Dice badge on predicted mask
        # axes[1].text(
        #     0.02, 0.02, f"Dice: {perImageDice:.3f}",
        #     transform=axes[1].transAxes, fontsize=14, fontweight="bold",
        #     color="white", va="bottom", ha="left",
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        # )

        # Coverage boxes above predicted and ground truth
        axes[0].text(
            0.5, 1.02, "Original Image",
            transform=axes[0].transAxes,
            fontsize=18.5, fontweight="bold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                    edgecolor="gray", alpha=0.9),
        )
        for ax, mask in [(axes[1], prediction), (axes[2], groundTruth)]:
            ax.text(
                0.5, 1.02, self.formatCoverage(mask),
                transform=ax.transAxes,
                fontsize=18.5, fontweight="bold", ha="center", va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor="gray", alpha=0.9),
            )

        fig.savefig(
            self.outputDirectory / f"Inference_{index + 1:03d}.png",
            dpi=600, bbox_inches="tight", facecolor="white",
        )
        fig.savefig(
            self.outputDirectory / f"Inference_{index + 1:03d}.pdf",
            dpi=600, bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)

    def plotBestWorst(self, imageScores: list[tuple[int, float]]) -> None:
        """Generate a combined figure showing best and worst predictions."""
        imageScores.sort(key=lambda x: x[1], reverse=True)
        best = imageScores[:3]
        worst = imageScores[-3:]

        fig, axes = plt.subplots(2, 6, figsize=(36, 13))
        fig.subplots_adjust(wspace=0.04, hspace=0.15)

        for row, (samples, label) in enumerate([(best, "Best"), (worst, "Worst")]):
            for col, (idx, dice) in enumerate(samples):
                image, groundTruth = self.dataset[idx]
                imageInput = np.expand_dims(image.numpy().astype(np.float32), axis=0)
                output = self.session.run(
                    None, {self.session.get_inputs()[0].name: imageInput}
                )[0]
                prediction = np.argmax(output, axis=1).squeeze()

                axImg = axes[row, col * 2]
                axPred = axes[row, col * 2 + 1]

                axImg.imshow(image.permute(1, 2, 0).numpy())
                axImg.set_title(f"Image #{idx + 1}", fontsize=14, fontweight="bold")
                axImg.axis("off")

                axPred.imshow(self.classIndicesToRGB(prediction))
                axPred.set_title(f"Dice: {dice:.3f}", fontsize=14, fontweight="bold")
                axPred.axis("off")

            # Row label
            axes[row, 0].text(
                -0.15,
                0.5,
                f"{label} Cases",
                transform=axes[row, 0].transAxes,
                fontsize=20,
                fontweight="bold",
                rotation=90,
                va="center",
                ha="center",
            )

        # Shared legend at bottom
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markerfacecolor=np.array(props["color"]) / 255,
            )
            for props in self.classColors.values()
        ]
        fig.legend(
            handles=handles,
            labels=CLASS_NAMES,
            loc="lower center",
            ncol=len(CLASS_NAMES),
            fontsize=15,
            frameon=True,
            edgecolor="gray",
            fancybox=True,
        )

        fig.savefig(
            self.outputDirectory / "best_worst_predictions.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            self.outputDirectory / "best_worst_predictions.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        print(f"Best/worst figure saved.")

    def plotConfusionMatrix(
        self, allPredictions: torch.Tensor, allGroundTruths: torch.Tensor
    ) -> None:
        """Generate a normalized confusion matrix figure."""
        cm = multiclass_confusion_matrix(
            allPredictions,
            allGroundTruths,
            num_classes=NUM_CLASSES_new,
            normalize="true",
        )
        cm = cm.numpy()

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks(range(NUM_CLASSES_new))
        ax.set_yticks(range(NUM_CLASSES_new))
        ax.set_xticklabels(CLASS_NAMES, fontsize=13, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES, fontsize=13)
        ax.set_xlabel("Predicted Class", fontsize=16, labelpad=10)
        ax.set_ylabel("True Class", fontsize=16, labelpad=10)
        ax.set_title(
            "Normalized Confusion Matrix", fontsize=18, fontweight="bold", pad=12
        )

        # Annotate cells
        for i in range(NUM_CLASSES_new):
            for j in range(NUM_CLASSES_new):
                color = "white" if cm[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=color,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        fig.savefig(
            self.outputDirectory / "confusion_matrix.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            self.outputDirectory / "confusion_matrix.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        print("Confusion matrix saved.")

    def printResultsTable(self, avgMetrics: dict) -> None:
        """Print a formatted results table for the paper."""
        print(f"\n{'=' * 70}")
        print(f"  TEST SET RESULTS")
        print(f"{'=' * 70}")

        print(f"\n  Macro-averaged metrics:")
        for key in SCALAR_KEYS:
            if key in avgMetrics:
                print(f"    {key:<25} {avgMetrics[key]:.4f}")

        print(f"\n  Per-class metrics:")
        header = f"    {'Metric':<25}" + "".join(f"{c:>15}" for c in CLASS_NAMES)
        print(header)
        print(f"    {'-' * (25 + 15 * len(CLASS_NAMES))}")
        for key in PER_CLASS_KEYS:
            if key in avgMetrics and avgMetrics[key] is not None:
                row = f"    {key:<25}"
                for v in avgMetrics[key]:
                    row += f"{v:>15.4f}"
                print(row)

        # LaTeX-ready table
        print(f"\n  LaTeX table row (macro):")
        vals = " & ".join(
            f"{avgMetrics[k]:.4f}" for k in SCALAR_KEYS if k in avgMetrics
        )
        print(f"    Model & {vals} \\\\")

    def runInference(self):
        aggregatedScalars = {key: 0.0 for key in SCALAR_KEYS}
        aggregatedPerClass = {key: None for key in PER_CLASS_KEYS}
        imageScores = []
        allPredictions = []
        allGroundTruths = []

        for i in range(len(self.dataset)):
            image, groundTruth = self.dataset[i]
            imageInput = np.expand_dims(image.numpy().astype(np.float32), axis=0)
            output = self.session.run(
                None, {self.session.get_inputs()[0].name: imageInput}
            )[0]

            outputTensor = from_numpy(output)
            gtTensor = groundTruth.unsqueeze(0)
            metrics = computeMetrics(outputTensor, gtTensor)

            # Accumulate scalars
            for key in SCALAR_KEYS:
                if key in metrics:
                    aggregatedScalars[key] += metrics[key]

            # Accumulate per-class
            for key in PER_CLASS_KEYS:
                if key in metrics and metrics[key] is not None:
                    if aggregatedPerClass[key] is None:
                        aggregatedPerClass[key] = [0.0] * len(metrics[key])
                    for j, v in enumerate(metrics[key]):
                        aggregatedPerClass[key][j] += v

            # Track per-image Dice for best/worst selection
            perImageDice = metrics.get(
                "Mean Dice", metrics.get("Dice Coefficient", 0.0)
            )
            imageScores.append((i, perImageDice))

            # Collect for confusion matrix
            predArgmax = torch.argmax(outputTensor, dim=1).squeeze()
            allPredictions.append(predArgmax)
            allGroundTruths.append(groundTruth)

            # Plot individual result
            prediction = np.argmax(output, axis=1).squeeze()
            self.plotResults(image, prediction, groundTruth, i, perImageDice)

        # Average metrics
        n = len(self.dataset)
        avgMetrics = {key: val / n for key, val in aggregatedScalars.items()}
        for key in PER_CLASS_KEYS:
            if aggregatedPerClass[key] is not None:
                avgMetrics[key] = [v / n for v in aggregatedPerClass[key]]
            else:
                avgMetrics[key] = None

        # Print results
        self.printResultsTable(avgMetrics)

        # Best/worst figure
        self.plotBestWorst(imageScores)

        # Confusion matrix
        allPred = torch.cat([p.unsqueeze(0) for p in allPredictions], dim=0)
        allGT = torch.cat([g.unsqueeze(0) for g in allGroundTruths], dim=0)
        self.plotConfusionMatrix(allPred, allGT)

    def formatCoverage(self, mask) -> str:
        coverage = self.calculateClassCoverage(mask)
        names = list(coverage.keys())
        # 2 rows, 2 classes per row
        row1 = f"{names[0]}: {coverage[names[0]]:.1f}%     {names[1]}: {coverage[names[1]]:.1f}%"
        row2 = f"{names[2]}: {coverage[names[2]]:.1f}%     {names[3]}: {coverage[names[3]]:.1f}%"
        return f"{row1}\n{row2}"
if __name__ == "__main__":
    model_path = Path(
        PROJECT_ROOT / "Trained models" / "SimpleUNet" / "modelTrial0.onnx"
    )
    device = setupDevice()
    output_directory = MODEL_PATH / "Predictions" / "SimpleUNet_revised"
    ModelTester(
        modelPath=model_path,
        rootPath=TESTING_PATH,
        outputDirectory=output_directory,
        device=device,
    )
    model_path = Path(
        PROJECT_ROOT / "Trained models" / "AttentionUNet" / "modelTrial0.onnx"
    )
    device = setupDevice()
    output_directory = MODEL_PATH / "Predictions" / "AttentionUNet_revised"
    ModelTester(
        modelPath=model_path,
        rootPath=TESTING_PATH,
        outputDirectory=output_directory,
        device=device,
    )

