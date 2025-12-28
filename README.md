# breast-cancer-detection-IDC
Explainable deep learning system for breast cancer detection using histopathology images (IDC dataset).

🩺 Breast Cancer Detection using Explainable Deep Learning (IDC Dataset)


📌 Overview

Breast cancer is one of the most common and life-threatening cancers worldwide.
Early and accurate detection plays a critical role in improving patient outcomes.

This project implements an end-to-end deep learning pipeline to classify breast histopathology image patches as benign or malignant using transfer learning and explainable AI techniques.
The focus is not only on performance, but also on medical reliability, evaluation correctness, and interpretability.


🎯 Problem Statement

Given a histopathology image patch extracted from a breast biopsy slide, the goal is to determine whether the tissue is:

Benign (non-cancerous)

Malignant (Invasive Ductal Carcinoma – IDC)

Since missing a malignant case can have severe clinical consequences, the project prioritizes recall for malignant samples over raw accuracy.


🧬 Dataset

Dataset Name: Breast Histopathology Images (IDC)

Source: Kaggle

Patients: 279

Image Type: Histopathology tissue patches (50×50 pixels)

Classes:

0 → Benign

1 → Malignant (IDC)

Folder Structure
IDC_regular_ps50_idx5/
├── Patient_ID/
│   ├── 0/   (benign patches)
│   └── 1/   (malignant patches)


⚠️ Important Dataset Handling Decision

A patient-wise split was used instead of a random image split to prevent data leakage.
This ensures that patches from the same patient never appear in both training and evaluation sets.


🧠 Methodology
Model Architecture

Backbone: DenseNet121 (pretrained on ImageNet)

Approach: Transfer Learning with frozen base

Classifier Head:

Global Average Pooling

Fully connected layer (ReLU)

Batch Normalization

Dropout (regularization)

Sigmoid output (binary classification)

Why DenseNet121?

DenseNet encourages feature reuse and performs well on texture-rich medical images, making it suitable for histopathology data.


⚙️ Training Strategy

Frozen backbone to reduce overfitting and computational cost

Binary Cross-Entropy loss

Adam optimizer

Class weighting to address class imbalance

Medically safe data augmentation (horizontal flips only)

Training performed on Google Colab (GPU)

Fine-tuning of the backbone was intentionally skipped due to compute constraints and to maintain training stability.


📊 Evaluation Strategy

Rather than relying solely on accuracy, evaluation focused on:

Recall (primary metric) – minimizing false negatives

Precision

F1-score

Confusion Matrix

ROC-AUC

This aligns with real-world medical decision-making, where missing cancer cases is more dangerous than false positives.


📈 Results
Confusion Matrix (Test Set)
Predicted Benign	Predicted Malignant
Actual Benign	23,431	6,178
Actual Malignant	1,185	9,868
Key Metrics (Malignant Class)

Recall: ~0.89

Precision: ~0.61

Accuracy: ~0.82

The model successfully identifies the majority of malignant cases while accepting a higher false-positive rate — a clinically safer trade-off.


🔍 Explainability – Grad-CAM

To improve transparency and trust, Grad-CAM (Gradient-weighted Class Activation Mapping) was applied to visualize which regions of histopathology patches influenced the model’s predictions.

Grad-CAM heatmaps confirm that the model focuses on relevant tissue structures rather than background noise.

Example Visualization

Explainability is a critical requirement for medical AI systems and was treated as a first-class component of this project.


⚠️ Limitations

Predictions are made at the patch level, not whole-slide images

No claim of clinical deployment or diagnostic replacement

Fine-tuning of the backbone was not performed

Dataset limited to IDC subtype only


🚀 Future Work

Fine-tuning the DenseNet backbone on GPU

Aggregating patch-level predictions for whole-slide inference

Multi-magnification analysis

Lightweight deployment as a web demo (e.g., Streamlit)


🧪 Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas

Scikit-learn

Matplotlib

OpenCV (Grad-CAM)


📌 Conclusion

This project demonstrates a complete, responsible, and explainable medical deep learning workflow, emphasizing correct dataset handling, clinically meaningful evaluation, and interpretability.
The goal was not just to build a model that performs well, but one that can be understood, trusted, and critically evaluated.

⚠️ Disclaimer

This project is intended for educational and research purposes only and is not a certified medical diagnostic tool.
