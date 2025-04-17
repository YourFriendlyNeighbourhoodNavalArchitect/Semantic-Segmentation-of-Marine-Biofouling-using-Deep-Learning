using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Production
{
    public partial class GUI : Form
    {
        private readonly ProgressBar progressBar;
        private readonly ToolStripStatusLabel statusLabel;
        private OpenCvSharp.Size imageResolution;
        private readonly Vec3b[] classColors;
        private CancellationTokenSource? cancellationTokenSource;

        private static readonly Color ButtonDefaultColor = Color.FromArgb(192, 192, 255);
        private static readonly Color ButtonCancelColor = Color.Red;
        private const string ButtonTextStart = "Initiate Predictions";
        private const string ButtonTextCancel = "Cancel";

        public GUI()
        {
            InitializeComponent();

            this.BackColor = Color.LightGray;
            this.Font = new Font("Segoe UI", 10);

            StatusStrip statusStrip = new();
            statusLabel = new ToolStripStatusLabel("Ready");
            statusStrip.Items.Add(statusLabel);
            this.Controls.Add(statusStrip);

            progressBar = new ProgressBar
            {
                Dock = DockStyle.Bottom,
                Maximum = 100,
                Step = 1,
                Style = ProgressBarStyle.Continuous
            };
            this.Controls.Add(progressBar);

            imageResolution = new OpenCvSharp.Size(304, 304);
            classColors =
            [
                new Vec3b(0, 255, 0),
                new Vec3b(106, 255, 255),
                new Vec3b(51, 87, 255),
                new Vec3b(177, 41, 157),
                new Vec3b(255, 138, 43)
            ];
        }

        private enum ProcessingState
        {
            Ready,
            Processing,
            Cancelled
        }

        private void UpdateProcessingState(ProcessingState state, string message = "")
        {
            if (InvokeRequired)
            {
                Invoke(() => UpdateProcessingState(state, message));
                return;
            }

            switch (state)
            {
                case ProcessingState.Ready:
                    StartPredictionButton.Text = ButtonTextStart;
                    StartPredictionButton.BackColor = ButtonDefaultColor;
                    statusLabel.Text = "Ready";
                    progressBar.Value = 0;
                    break;

                case ProcessingState.Processing:
                    StartPredictionButton.Text = ButtonTextCancel;
                    StartPredictionButton.BackColor = ButtonCancelColor;
                    statusLabel.Text = "Processing...";
                    break;

                case ProcessingState.Cancelled:
                    StartPredictionButton.Text = ButtonTextStart;
                    StartPredictionButton.BackColor = ButtonDefaultColor;
                    statusLabel.Text = "Operation cancelled";
                    break;
            }
        }

        private void UpdateProgress(int progress)
        {
            if (InvokeRequired)
            {
                Invoke(() => UpdateProgress(progress));
                return;
            }
            progressBar.Value = Math.Min(100, Math.Max(0, progress));
        }

        private void ShowError(string message)
        {
            if (InvokeRequired)
            {
                Invoke(() => ShowError(message));
                return;
            }
            MessageBox.Show(message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private void SelectInputFolderButton_Click(object sender, EventArgs e)
        {
            using FolderBrowserDialog fbd = new();
            if (fbd.ShowDialog() == DialogResult.OK)
            {
                InputFolderTextBox.Text = fbd.SelectedPath;
            }
        }

        private void SelectOutputFolderButton_Click(object sender, EventArgs e)
        {
            using FolderBrowserDialog fbd = new();
            if (fbd.ShowDialog() == DialogResult.OK)
            {
                OutputFolderTextBox.Text = fbd.SelectedPath;
            }
        }

        private void SelectModelButton_Click(object sender, EventArgs e)
        {
            using OpenFileDialog ofd = new()
            {
                Filter = "ONNX files (*.onnx)|*.onnx"
            };
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                ModelPathTextBox.Text = ofd.FileName;
            }
        }

        private async void StartPredictionButton_Click(object sender, EventArgs e)
        {
            if (StartPredictionButton.Text == ButtonTextCancel)
            {
                cancellationTokenSource?.Cancel();
                return;
            }

            if (!ValidateInputs(out string inputFolder, out string outputFolder, out string modelPath))
            {
                return;
            }

            using (cancellationTokenSource = new CancellationTokenSource())
            {
                try
                {
                    UpdateProcessingState(ProcessingState.Processing);
                    await Task.Run(() => PredictMasks(modelPath, inputFolder, outputFolder, cancellationTokenSource.Token));
                    MessageBox.Show("Predictions complete! Masks saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (OperationCanceledException)
                {
                    UpdateProcessingState(ProcessingState.Cancelled);
                    MessageBox.Show("Process cancelled.", "Cancelled", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    ShowError($"Processing error: {ex.Message}");
                }
                finally
                {
                    UpdateProcessingState(ProcessingState.Ready);
                }
            }
        }

        private void PredictMasks(string modelPath, string inputFolder, string outputFolder, CancellationToken cancellationToken)
        {
            using var session = CreateInferenceSession(modelPath);
            var loadedImages = LoadImages(inputFolder);
            if (loadedImages.Count == 0)
            {
                throw new InvalidOperationException("No valid images found in the input folder.");
            }

            int totalFiles = loadedImages.Count;
            int processedFiles = 0;

            foreach (var (filePath, processedImage) in loadedImages)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    ProcessImage(session, processedImage, filePath, outputFolder);
                    processedFiles++;
                    UpdateProgress((int)((processedFiles / (double)totalFiles) * 100));
                }
                catch (Exception ex)
                {
                    ShowError($"Error processing {Path.GetFileName(filePath)}: {ex.Message}");
                }
                finally
                {
                    processedImage.Dispose();
                }
            }
        }

        private static InferenceSession CreateInferenceSession(string modelPath)
        {
            SessionOptions options = new();
            options.AppendExecutionProvider_CPU();
            return new InferenceSession(modelPath, options);
        }

        private void ProcessImage(InferenceSession session, Mat processedImage, string filePath, string outputFolder)
        {
            var inputTensor = CreateInputTensor(processedImage);
            using var prediction = RunInference(session, inputTensor);
            using var RGBMask = ClassIndicesToRGB(prediction);
            string outputFilePath = Path.Combine(outputFolder, Path.GetFileNameWithoutExtension(filePath) + "_mask.png");
            Cv2.ImWrite(outputFilePath, RGBMask);
        }

        private DenseTensor<float> CreateInputTensor(Mat processedImage)
        {
            DenseTensor<float> inputTensor = new([1, 3, imageResolution.Height, imageResolution.Width]);
            for (int y = 0; y < processedImage.Rows; y++)
            {
                for (int x = 0; x < processedImage.Cols; x++)
                {
                    Vec3f pixel = processedImage.At<Vec3f>(y, x);
                    inputTensor[0, 0, y, x] = pixel.Item0;
                    inputTensor[0, 1, y, x] = pixel.Item1;
                    inputTensor[0, 2, y, x] = pixel.Item2;
                }
            }

            return inputTensor;
        }

        private Mat RunInference(InferenceSession session, DenseTensor<float> inputTensor)
        {
            string inputName = session.InputMetadata.Keys.First();
            var namedInput = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
            using var results = session.Run([namedInput]);
            var output = results[0].AsTensor<float>();
            Mat prediction = new(imageResolution.Height, imageResolution.Width, MatType.CV_8UC1);
            float[] outputArray = [.. output];

            for (int i = 0; i < imageResolution.Height; i++)
            {
                for (int j = 0; j < imageResolution.Width; j++)
                {
                    int baseIndex = (i * imageResolution.Width + j);
                    float[] pixelProbs = new float[classColors.Length];
                    for (int c = 0; c < classColors.Length; c++)
                    {
                        pixelProbs[c] = outputArray[baseIndex + c * imageResolution.Height * imageResolution.Width];
                    }
                    prediction.Set(i, j, (byte)Array.IndexOf(pixelProbs, pixelProbs.Max()));
                }
            }

            return prediction;
        }

        private List<(string FilePath, Mat Image)> LoadImages(string inputFolder)
        {
            List<(string FilePath, Mat Image)> loadedImages = [];
            HashSet<string> allowedExtensions = new(StringComparer.OrdinalIgnoreCase) { ".jpg", ".png" };
            var inputFiles = Directory.EnumerateFiles(inputFolder)
                                      .Where(file => allowedExtensions.Contains(Path.GetExtension(file))).ToArray();

            foreach (string filePath in inputFiles)
            {
                try
                {
                    using Mat originalImage = Cv2.ImRead(filePath);
                    if (originalImage.Empty())
                    {
                        ShowError($"Image '{Path.GetFileName(filePath)}' is empty.");
                        continue;
                    }

                    using Mat resizedImage = new();
                    Cv2.Resize(originalImage, resizedImage, imageResolution, 0, 0, InterpolationFlags.Linear);
                    using Mat RGBImage = new();
                    Cv2.CvtColor(resizedImage, RGBImage, ColorConversionCodes.BGR2RGB);
                    Mat processedImage = new();
                    RGBImage.ConvertTo(processedImage, MatType.CV_32FC3, 1.0 / 255.0);
                    loadedImages.Add((filePath, processedImage));
                }
                catch (Exception ex)
                {
                    ShowError($"Error loading '{Path.GetFileName(filePath)}': {ex.Message}");
                }
            }

            return loadedImages;
        }

        private Mat ClassIndicesToRGB(Mat prediction)
        {
            if (prediction.Type() != MatType.CV_8UC1)
            {
                prediction.ConvertTo(prediction, MatType.CV_8UC1);
            }

            Mat RGBMask = new(prediction.Rows, prediction.Cols, MatType.CV_8UC3, new Scalar(0, 0, 0));
            for (int y = 0; y < prediction.Rows; y++)
            {
                for (int x = 0; x < prediction.Cols; x++)
                {
                    byte classIndex = prediction.At<byte>(y, x);
                    Vec3b color = (classIndex < classColors.Length) ?
                        classColors[classIndex] : new Vec3b(0, 0, 0);
                    RGBMask.Set(y, x, color);
                }
            }

            return RGBMask;
        }

        private bool ValidateInputs(out string inputFolder, out string outputFolder, out string modelPath)
        {
            inputFolder = InputFolderTextBox.Text;
            outputFolder = OutputFolderTextBox.Text;
            modelPath = ModelPathTextBox.Text;

            if (string.IsNullOrEmpty(inputFolder) || string.IsNullOrEmpty(outputFolder) ||
                string.IsNullOrEmpty(modelPath))
            {
                ShowError("Please provide input/output folders and a model path.");
                return false;
            }

            if (!Directory.Exists(inputFolder))
            {
                ShowError("Input folder does not exist.");
                return false;
            }

            if (!Directory.Exists(outputFolder))
            {
                try
                {
                    Directory.CreateDirectory(outputFolder);
                }
                catch (Exception ex)
                {
                    ShowError($"Could not create output folder: {ex.Message}");
                    return false;
                }
            }

            if (!File.Exists(modelPath))
            {
                ShowError("Model file does not exist.");
                return false;
            }

            return true;
        }
    }
}