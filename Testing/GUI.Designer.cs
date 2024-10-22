namespace Production
{
    partial class GUI
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            SelectInputFolderButton = new Button();
            SelectOutputFolderButton = new Button();
            SelectModelButton = new Button();
            StartPredictionButton = new Button();
            InputFolderTextBox = new TextBox();
            OutputFolderTextBox = new TextBox();
            ModelPathTextBox = new TextBox();
            SuspendLayout();

            SelectInputFolderButton.FlatAppearance.BorderColor = Color.Black;
            SelectInputFolderButton.Location = new Point(200, 75);
            SelectInputFolderButton.Name = "SelectInputFolderButton";
            SelectInputFolderButton.Size = new Size(200, 40);
            SelectInputFolderButton.TabIndex = 0;
            SelectInputFolderButton.Text = "Select Input Folder";
            SelectInputFolderButton.UseVisualStyleBackColor = true;
            SelectInputFolderButton.Click += SelectInputFolderButton_Click;

            SelectOutputFolderButton.Location = new Point(200, 195);
            SelectOutputFolderButton.Name = "SelectOutputFolderButton";
            SelectOutputFolderButton.Size = new Size(200, 40);
            SelectOutputFolderButton.TabIndex = 1;
            SelectOutputFolderButton.Text = "Select Output Folder";
            SelectOutputFolderButton.UseVisualStyleBackColor = true;
            SelectOutputFolderButton.Click += SelectOutputFolderButton_Click;

            SelectModelButton.Location = new Point(200, 315);
            SelectModelButton.Name = "SelectModelButton";
            SelectModelButton.Size = new Size(200, 40);
            SelectModelButton.TabIndex = 2;
            SelectModelButton.Text = "Provide Model Path";
            SelectModelButton.UseVisualStyleBackColor = true;
            SelectModelButton.Click += SelectModelButton_Click;

            StartPredictionButton.BackColor = Color.FromArgb(192, 192, 255);
            StartPredictionButton.FlatAppearance.BorderColor = Color.Black;
            StartPredictionButton.FlatAppearance.BorderSize = 2;
            StartPredictionButton.FlatStyle = FlatStyle.Flat;
            StartPredictionButton.Font = new Font("Segoe UI", 9F, FontStyle.Bold);
            StartPredictionButton.Location = new Point(200, 430);
            StartPredictionButton.Name = "StartPredictionButton";
            StartPredictionButton.Size = new Size(200, 50);
            StartPredictionButton.TabIndex = 3;
            StartPredictionButton.Text = "Initiate Predictions";
            StartPredictionButton.UseVisualStyleBackColor = false;
            StartPredictionButton.Click += StartPredictionButton_Click;

            InputFolderTextBox.Location = new Point(125, 30);
            InputFolderTextBox.Name = "InputFolderTextBox";
            InputFolderTextBox.Size = new Size(350, 27);
            InputFolderTextBox.TabIndex = 4;

            OutputFolderTextBox.Location = new Point(125, 150);
            OutputFolderTextBox.Name = "OutputFolderTextBox";
            OutputFolderTextBox.Size = new Size(350, 27);
            OutputFolderTextBox.TabIndex = 5;

            ModelPathTextBox.Location = new Point(125, 270);
            ModelPathTextBox.Name = "ModelPathTextBox";
            ModelPathTextBox.Size = new Size(350, 27);
            ModelPathTextBox.TabIndex = 6;

            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(582, 553);
            Controls.Add(ModelPathTextBox);
            Controls.Add(OutputFolderTextBox);
            Controls.Add(InputFolderTextBox);
            Controls.Add(StartPredictionButton);
            Controls.Add(SelectModelButton);
            Controls.Add(SelectOutputFolderButton);
            Controls.Add(SelectInputFolderButton);
            Name = "GUI";
            Text = "Marine Biofouling Segmentation";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button SelectInputFolderButton;
        private Button SelectOutputFolderButton;
        private Button SelectModelButton;
        private Button StartPredictionButton;
        private TextBox InputFolderTextBox;
        private TextBox OutputFolderTextBox;
        private TextBox ModelPathTextBox;
    }
}
