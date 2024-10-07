This project was carried out as part of the author's MSc thesis "Image Segmentation and Classification of Marine Biofouling Images with Attention U-Net". Aim of the work is to classify and segment images of marine surfaces using an advanced Attention U-Net architecture. This architecture enhances the traditional U-Net by incorporating attention gates to focus on the relevant regions of the input images. This allows the model to learn to concentrate on areas of significant importance while ignoring irrelevant background noise.

Marine biofouling is the accumulation of organisms like algae, barnacles, and other marine life on submerged surfaces. Monitoring and managing biofouling is essential for the shipping industry, offshore platforms, and aquaculture to reduce drag, fuel consumption, and the spread of invasive species. This project applies deep learning for automating the process of segmentation and classification thereof.

The model facilitates multi-class segmentation of each image into different classes: no fouling, light fouling, heavy fouling, and sea background.

The trained models are saved in ONNX format for easy deployment.

Live metric visualization (training and validation loss, dice coefficient and IoU score) is available in real-time during training for monitoring progress.

Optuna is used for hyperparameter tuning to ensure the model achieves optimal performance.
