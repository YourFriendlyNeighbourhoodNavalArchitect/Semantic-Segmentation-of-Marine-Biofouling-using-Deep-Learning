using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Production
{
    public partial class GUI : Form
    {
        private ProgressBar progressBar;
        private ToolStripStatusLabel statusLabel;

        public GUI()
        {
            InitializeComponent();
            this.BackColor = Color.LightGray;
            this.Font = new Font("Segoe UI", 10);

            // Add status bar
            StatusStrip statusStrip = new StatusStrip();
            statusLabel = new ToolStripStatusLabel("Ready");
            statusStrip.Items.Add(statusLabel);
            this.Controls.Add(statusStrip);

            // Add progress bar
            progressBar = new ProgressBar
            {
                Dock = DockStyle.Bottom,
                Maximum = 100,
                Step = 1,
                Style = ProgressBarStyle.Continuous
            };
            this.Controls.Add(progressBar);
        }

        private void SelectInputFolderButton_Click(object sender, EventArgs e)
        {
            using var fbd = new FolderBrowserDialog();
            if (fbd.ShowDialog() == DialogResult.OK)
            {
                InputFolderTextBox.Text = fbd.SelectedPath;
            }
        }

        private void SelectOutputFolderButton_Click(object sender, EventArgs e)
        {
            using var fbd = new FolderBrowserDialog();
            if (fbd.ShowDialog() == DialogResult.OK)
            {
                OutputFolderTextBox.Text = fbd.SelectedPath;
            }
        }

        private void SelectModelButton_Click(object sender, EventArgs e)
        {
            using var ofd = new OpenFileDialog();
            ofd.Filter = "ONNX files (*.onnx)|*.onnx";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                ModelPathTextBox.Text = ofd.FileName;
            }
        }

        private async void StartPredictionButton_Click(object sender, EventArgs e)
        {
            string inputFolder = InputFolderTextBox.Text;
            string outputFolder = OutputFolderTextBox.Text;
            string modelPath = ModelPathTextBox.Text;

            // Check if input folder, output folder, and model path are provided.
            if (string.IsNullOrEmpty(inputFolder) || string.IsNullOrEmpty(outputFolder) || string.IsNullOrEmpty(modelPath))
            {
                MessageBox.Show("Please provide input/output folders and a model path.", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // Disable the Start button to prevent re-clicking during the process
            StartPredictionButton.Enabled = false;

            // Update status and reset the progress bar
            statusLabel.Text = "Processing...";
            progressBar.Value = 0;

            // Run the prediction asynchronously
            await Task.Run(() => PredictMasks(modelPath, inputFolder, outputFolder));

            // Enable the Start button after completion
            StartPredictionButton.Enabled = true;
            statusLabel.Text = "Ready";
        }

        private void PredictMasks(string modelPath, string inputFolder, string outputFolder)
        {
            // Create a single inference session to improve performance.
            var options = new SessionOptions();
            options.AppendExecutionProvider_CPU();
            using var session = new InferenceSession(modelPath, options);

            // Fetch valid input files (JPG/PNG).
            var inputFiles = Directory.GetFiles(inputFolder, "*.jpg").Concat(Directory.GetFiles(inputFolder, "*.png")).ToArray();

            if (inputFiles.Length == 0)
            {
                MessageBox.Show("No valid image files found in the selected input folder.", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            foreach (var filePath in inputFiles)
            {
                try
                {
                    // Load the image using OpenCV.
                    using var image = Cv2.ImRead(filePath);
                    if (image.Empty())
                    {
                        throw new Exception("Failed to read the image.");
                    }

                    // Resize image to 256x256 (without preserving aspect ratio).
                    var resizedImage = new Mat();
                    Cv2.Resize(image, resizedImage, new OpenCvSharp.Size(256, 256), 0, 0, InterpolationFlags.Linear);

                    // Convert OpenCV image to tensor.
                    // OpenCV stores images as BGR by default, so we reorder it to RGB.
                    var inputTensor = new DenseTensor<float>(new[] { 1, 3, 256, 256 });
                    for (int y = 0; y < resizedImage.Rows; y++)
                    {
                        for (int x = 0; x < resizedImage.Cols; x++)
                        {
                            Vec3b pixel = resizedImage.At<Vec3b>(y, x);
                            // Normalize pixel values to [0, 1].
                            inputTensor[0, 0, y, x] = pixel.Item2 / 255.0f;
                            inputTensor[0, 1, y, x] = pixel.Item1 / 255.0f;
                            inputTensor[0, 2, y, x] = pixel.Item0 / 255.0f;
                        }
                    }

                    // Run inference using ONNX model.
                    var inputName = session.InputMetadata.Keys.First();
                    var namedInput = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
                    var results = session.Run(new[] { namedInput });

                    // Extract the output and apply argmax to get the predicted class for each pixel.
                    var output = results.First().AsTensor<float>();
                    var outputArray = output.ToArray();
                    var prediction = new Mat(256, 256, MatType.CV_8UC1);

                    for (int i = 0; i < 256; i++)
                    {
                        for (int j = 0; j < 256; j++)
                        {
                            // Get the pixel class probabilities.
                            int baseIndex = (i * 256 + j);
                            float[] pixelProbs = new float[4];
                            for (int c = 0; c < 4; c++)
                            {
                                pixelProbs[c] = outputArray[baseIndex + c * 65536];
                            }

                            // Find the class with the maximum probability.
                            int classIndex = Array.IndexOf(pixelProbs, pixelProbs.Max());
                            prediction.Set(i, j, (byte)classIndex);
                        }
                    }

                    // Convert the class indices to an RGB mask.
                    var RGBMask = ClassIndicesToRGB(prediction);
                    string outputFilePath = Path.Combine(outputFolder, Path.GetFileNameWithoutExtension(filePath) + "_mask.png");

                    // Save the mask image.
                    Cv2.ImWrite(outputFilePath, RGBMask);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error processing {Path.GetFileName(filePath)}: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }

            MessageBox.Show("Predictions complete! Masks saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        // Map class indices to BGR colors.
        private Mat ClassIndicesToRGB(Mat prediction)
        {
            Mat RGBMask = new Mat(prediction.Size(), MatType.CV_8UC3);

            for (int i = 0; i < prediction.Rows; i++)
            {
                for (int j = 0; j < prediction.Cols; j++)
                {
                    int classIndex = prediction.Get<byte>(i, j);
                    RGBMask.Set(i, j, classIndex switch
                    {
                        0 => new Vec3b(0, 255, 0),       // Green for class 0
                        1 => new Vec3b(102, 255, 255),   // Light Yellow for class 1 (BGR)
                        2 => new Vec3b(0, 0, 255),       // Red for class 2 (BGR)
                        3 => new Vec3b(255, 0, 0),       // Blue for class 3 (BGR)
                        _ => new Vec3b(0, 0, 0)          // Default to black
                    });
                }
            }

            return RGBMask;
        }
    }
}