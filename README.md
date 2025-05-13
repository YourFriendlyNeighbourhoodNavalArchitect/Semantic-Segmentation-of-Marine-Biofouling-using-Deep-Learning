# Semantic Segmentation of Marine Biofouling using Deep Learning

This repository contains the work related to the semantic segmentation of marine biofouling images using deep learning, specifically focusing on Attention U-Net architectures. The project progressed through two main stages, each available in its respective folder.

## Project Overview

Marine biofouling, the accumulation of organisms on submerged surfaces, poses significant challenges to maritime industries. This project aims to automate the detection and classification of biofouling using advanced deep learning techniques. The core methodology involves an Attention U-Net model, which enhances standard U-Net architectures by incorporating attention mechanisms to focus on salient image regions, thereby improving segmentation accuracy for multiple biofouling classes.

**Key Problem:** Automating the monitoring and management of marine biofouling to mitigate its negative impacts (increased drag, fuel consumption, spread of invasive species) in sectors like shipping, offshore energy, and aquaculture.

**Approach:** Utilizing an Attention U-Net for multi-class semantic segmentation (5 classes).

---

## Project Stages and Repository Structure

This project has evolved over time. Please navigate to the relevant folder based on your interest:

### 1. Diploma Thesis Version

* **[Link to Folder](./Diploma-Thesis/)**
* **Description:** This folder contains the initial implementation developed as part of the MSc Diploma thesis titled "Semantic Segmentation of Marine Biofouling Images with Attention U-Net".
* **Core Features:**
    * Implementation of the Attention U-Net architecture.
    * Multi-class segmentation into 5 distinct classes.
    * Basic metric logging and visualization during training.
    * Hyperparameter tuning using Optuna.
    * Model saving in ONNX format.
    * A simple Windows application prototype for deployment.

### 2. Marine 2025 Version

* **[Link to Folder](./Marine-2025/)**
* **Description:** This folder contains an enhanced and refined version of the project, building upon the foundational work of the Diploma thesis, as part of Marine 2025 Conference publication titled "A Novel Application of Attention U-Net for Marine Biofouling Classification and Segmentation".
* **Improvements over Stage 1:**
    * Refined architecture for improved accuracy.
    * Expanded dataset and elaborate data augmentation techniques.
    * Advanced hyperparameter tuning algorithm.
---

## General Information

* **Citation:** If you use this work, please cite the relevant stage.
* **License:** This project is licensed under the **MIT License**. See the LICENSE file for details.
* **Contact:**
    * Ioannis Karlatiras: `giannhskarlathras@gmail.com`
