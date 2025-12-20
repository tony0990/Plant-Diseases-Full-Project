# Plant-Diseases-Full-Project
# 🌿 Plant Disease Classification using Deep Learning

This project implements a **plant disease classification system** using **transfer learning** with multiple state-of-the-art convolutional neural networks.  
The models are trained on the **PlantVillage dataset** and evaluated using standard classification metrics.  
The final optimized model is also converted to **TensorFlow Lite** for **mobile deployment**.

---

## 📌 Project Objectives

- Classify plant leaf images into disease categories
- Compare the performance of multiple pretrained CNN architectures
- Apply data augmentation and fine-tuning to improve generalization
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix
- Explain model decisions using **Grad-CAM**
- Optimize the best model for **mobile and edge devices** using **TensorFlow Lite**

---

## 📂 Dataset Description

**Dataset:** PlantVillage  
**Source:** Publicly available plant disease dataset  

### Characteristics:
- RGB leaf images
- Multiple plant species and disease categories
- Images captured under controlled conditions

### Dataset Variants Used:
- Color images
- Grayscale images
- Segmented leaf images (background removed)

### Data Splitting Strategy:
- **70% Training**
- **15% Validation**
- **15% Testing**
- Stratified splitting was applied to preserve class balance
- Splitting was performed at the **file-path level** to prevent data leakage

---

## 🧠 Models Used

Five deep learning models were implemented and compared:

1. **ResNet50**
2. **EfficientNetB3**
3. **MobileNetV2**
4. **DenseNet121**
5. **InceptionV3**

All models were trained using **transfer learning** with pretrained ImageNet weights.

---

## 🔍 Model Descriptions

### 🔹 ResNet50
- Uses **residual (skip) connections** to enable deep networks
- Mitigates vanishing gradient problem
- 50-layer deep convolutional network
- Excellent feature extraction capability

**Pretraining:** ImageNet (1.2M images, 1000 classes)

---

### 🔹 EfficientNetB3
- Uses **compound scaling** (depth, width, resolution)
- Achieves high accuracy with fewer parameters
- Efficient and well-balanced architecture

**Pretraining:** ImageNet  
**Key Advantage:** High accuracy with optimized computation

---

### 🔹 MobileNetV2
- Designed specifically for **mobile and edge devices**
- Uses **depthwise separable convolutions**
- Very lightweight and fast

**Pretraining:** ImageNet  
**Key Advantage:** Small model size and fast inference

---

### 🔹 DenseNet121
- Uses **dense connections** between layers
- Encourages feature reuse
- Reduces number of parameters while maintaining performance

**Pretraining:** ImageNet  
**Key Advantage:** Strong gradient flow and feature propagation

---

### 🔹 InceptionV3
- Uses **parallel convolution paths** with different kernel sizes
- Captures multi-scale spatial features
- Computationally efficient despite depth

**Pretraining:** ImageNet  
**Key Advantage:** Multi-scale feature extraction

---

## 🏋️ Training Strategy

- Input image size: **224 × 224**
- Data augmentation applied during training:
  - Random flip
  - Random rotation
  - Random zoom
  - Random contrast
- Two-stage training:
  1. **Feature extraction** (freeze backbone)
  2. **Fine-tuning** (unfreeze top layers)
- Optimizer: Adam
- Learning rate scheduling: ReduceLROnPlateau
- Early stopping and model checkpointing applied

---

## 📊 Evaluation Metrics

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best-performing model was selected based on **validation and test accuracy**.

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM was applied to visualize **which regions of the leaf images influenced model predictions**.  
This improves model transparency and interpretability, especially for disease detection.

---

## 📱 Mobile Deployment (TensorFlow Lite)

The final trained model was converted to **TensorFlow Lite**:

### Model Formats:
- FP32 TFLite
- Dynamic Range Quantized TFLite

### Benefits:
- Model size reduced from ~200 MB to ~22 MB
- Faster inference
- Suitable for Android and iOS deployment

---

## 📁 Repository Structure

