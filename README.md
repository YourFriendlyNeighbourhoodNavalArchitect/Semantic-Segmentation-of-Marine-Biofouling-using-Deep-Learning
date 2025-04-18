# Semantic Segmentation of Marine Biofouling Images with Attention U-Net

This repository contains the work related to the semantic segmentation of marine biofouling images using deep learning, specifically focusing on Attention U-Net architectures. The project progressed through two main stages, each available in its respective folder.

## Project Overview

Marine biofouling, the accumulation of organisms on submerged surfaces, poses significant challenges to maritime industries. This project aims to automate the detection and classification of biofouling using advanced deep learning techniques. The core methodology involves an Attention U-Net model, which enhances standard U-Net architectures by incorporating attention mechanisms to focus on salient image regions, thereby improving segmentation accuracy for multiple biofouling classes.

**Key Problem:** Automating the monitoring and management of marine biofouling to mitigate its negative impacts (increased drag, fuel consumption, spread of invasive species) in sectors like shipping, offshore energy, and aquaculture.

**Approach:** Utilizing an Attention U-Net for multi-class semantic segmentation (5 classes) of marine surface images.

---

## Project Stages & Repository Structure

This project has evolved over time. Please navigate to the relevant folder based on your interest:

### 1. Stage 1: MSc Thesis Version (`./msc-thesis-version/`)

* **[Link to Folder](./msc-thesis-version/)**
* **Description:** This folder contains the initial implementation developed as part of the MSc thesis titled "Semantic Segmentation of Marine Biofouling Images with Attention U-Net".
* **Core Features:**
    * Implementation of the Attention U-Net architecture.
    * Multi-class segmentation into 5 distinct classes.
    * Basic metric logging and visualization during training.
    * Hyperparameter tuning using Optuna.
    * Model saving in ONNX format.
    * A simple Windows application prototype for deployment.

### 2. Stage 2: Marine 2025 Version (`./paper-publication-version/`)

* **[Link to Folder](./Marine 2025/)**
* **Description:** This folder contains an enhanced and refined version of the project, developed subsequently for the conference publication. It builds upon the foundational work of the MSc thesis.
* **Improvements over Stage 1 (Examples - *Please update with your actual improvements*):**
    * [e.g., Refined Attention U-Net architecture for improved accuracy]
    * [e.g., Expanded dataset or different data augmentation techniques]
    * [e.g., More comprehensive evaluation metrics and comparative analysis]
    * [e.g., Optimizations for faster inference or training]
    * [e.g., Updated deployment mechanism or user interface]
* **Publication:** [*(Optional)* Add citation details here once available: *Author(s). (Year). Title of Paper. Journal/Conference Name.*]

---

## General Information

* **Citation:** If you use this work, please cite the relevant stage (MSc Thesis or Paper Publication - *add details as appropriate*).
* **License:** [Specify Your License, e.g., MIT, Apache 2.0]
* **Contact:** [Your Name/Email/GitHub Profile Link]
