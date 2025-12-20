## 🧹 Data Preprocessing Summary

Effective data preprocessing is a critical step to ensure model robustness, prevent data leakage, and improve generalization. The following preprocessing steps were applied consistently across all models used in this project.

---

### 🔹 Image Loading and Resizing
- All images were loaded as **RGB images**.
- Images were resized to a configurable resolution:
  - **224 × 224** (default, optimized for pretrained ImageNet models)
  - **256 × 256** (optional, for higher spatial detail)
- The input size can be selected based on the trade-off between **accuracy and computational cost**.
- Using standard input sizes ensures compatibility with pretrained ImageNet weights.

---

### 🔹 Label Encoding
- Class labels were encoded as **integer indices**.
- A consistent class order was maintained across training, validation, testing, and inference.
- This guarantees correct class-to-index mapping during evaluation and deployment.

---

### 🔹 Dataset Splitting
- The dataset was split programmatically using file-path based sampling:
  - **70% Training**
  - **15% Validation**
  - **15% Testing**
- **Stratified splitting** was applied to preserve class distribution across all subsets.
- The split was performed **before dataset creation** to prevent data leakage.

---

### 🔹 Data Augmentation (Training Only)
Data augmentation was applied **only to the training set** to improve generalization and reduce overfitting.

Applied augmentations include:
- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast adjustment

Augmentation layers were embedded **inside the model architecture**, ensuring:
- Automatic disabling during validation and testing
- Consistent preprocessing during inference

---

### 🔹 Model-Specific Preprocessing
- Each pretrained architecture uses a specific input normalization strategy.
- The appropriate `preprocess_input` function from `tf.keras.applications` was applied.
- This aligns the input image distribution with the ImageNet pretraining regime.
- Correct preprocessing is essential for effective transfer learning.

---

### 🔹 Performance Optimization
- Images were converted to `float32` format.
- TensorFlow `tf.data` pipelines were used for:
  - Efficient batching
  - Parallel data loading
  - Prefetching
- These optimizations reduce training time and improve hardware utilization.

---

### 🔹 Consistency Across Training and Deployment
- All preprocessing steps were preserved inside the model graph.
- This guarantees identical behavior during:
  - Training
  - Validation
  - Testing
  - TensorFlow Lite inference on mobile devices

---

### ✅ Summary
The preprocessing pipeline provides flexibility in input resolution (224×224 or 256×256), prevents data leakage, improves model generalization, and ensures reproducible performance across different environments.

