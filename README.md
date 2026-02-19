# 🌿 Comparative Analysis of Deep CNN Architectures for Plant Disease Detection

## 📌 Overview

This project presents a comparative study of four Convolutional Neural Network (CNN) architectures for plant disease detection using the **PlantVillage dataset**.

The models evaluated include:

* VGG19
* Mini-AlexNet
* Mini-GoogLeNet
* MobileNetV2 (Fast)

The goal of this study was to analyze the trade-off between **accuracy, training time, model size, and deployment feasibility**, and determine the most practical architecture for real-world agricultural applications.

---

## 🎯 Objective

Plant diseases significantly impact crop yield and food security. Traditional disease detection methods are manual, time-consuming, and inconsistent.

This project aims to:

* Build deep learning models for multi-class plant disease classification.
* Compare deep vs lightweight CNN architectures.
* Identify the most efficient model for real-time mobile deployment.
* Analyze accuracy, speed, and computational efficiency.

---

## 📊 Dataset

* **Dataset:** PlantVillage
* **Total Images:** 54,000+
* **Classes:** 38 (healthy + diseased plant leaves)
* **Split:** 80% Training | 10% Validation | 10% Testing

Images were resized to 256×256 and augmented using:

* Rotation
* Horizontal & Vertical Flip
* Zoom
* Brightness variation

---

## 🧠 Models Implemented

### 1️⃣ VGG19 (Transfer Learning)

* Pre-trained on ImageNet
* Deep architecture (19 layers)
* Large parameter size
* Slower convergence

### 2️⃣ Mini-AlexNet

* Simplified version of AlexNet
* Trained from scratch
* Balanced performance

### 3️⃣ Mini-GoogLeNet

* Inception-inspired architecture
* Multi-scale feature extraction
* Highest raw validation accuracy

### 4️⃣ MobileNetV2 (Fast) ⭐

* Depthwise separable convolutions
* Inverted residuals
* Lightweight and fast
* Best overall performance

---

## 🏆 Results Summary

| Model                  | Validation Accuracy | Training Time | Model Size |
| ---------------------- | ------------------- | ------------- | ---------- |
| VGG19                  | 78.12%              | ~50 min       | ~549 MB    |
| Mini-AlexNet           | 93.78%              | ~45 min       | ~230 MB    |
| Mini-GoogLeNet         | 95.09%              | ~55 min       | ~265 MB    |
| **MobileNetV2 (Fast)** | **90.18%**          | **~25 min**   | **~45 MB** |

### 🔥 Final Conclusion

Although Mini-GoogLeNet achieved the highest raw accuracy (95.09%), **MobileNetV2 (Fast)** delivered the best overall balance between:

* Accuracy
* Speed
* Memory efficiency
* Real-world deployability

MobileNetV2 (Fast) was identified as the most suitable architecture for **mobile and edge-based plant disease detection systems**.

---

## 📈 Evaluation Metrics

* Accuracy
* Cross-Entropy Loss
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training Time
* Model Size

---

## 🛠 Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab (GPU: Tesla T4)

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Notebook

Open the `.ipynb` file in:

* Google Colab
  or
* Jupyter Notebook

Make sure your dataset directory structure is:

```
dataset/
 ├── train/
 ├── val/
 └── test/
```

---

## 📂 Project Structure

```
📁 Plant-Disease-Detection
│
├── 📓 notebooks/
│   └── model_training.ipynb
│
├── 📊 results/
│   ├── comparison_plots.png
│   └── confusion_matrix.png
│
├── 📁 models/
│   ├── mini_googlenet_best.h5
│   └── mobilenetv2_fast_best.h5
│
├── README.md
└── requirements.txt
```

---

## 📌 Key Learnings

* Deeper models do not always guarantee better performance.
* Lightweight architectures can achieve competitive accuracy.
* Model efficiency is critical for agricultural deployment.
* MobileNetV2 provides an optimal trade-off between performance and computational cost.

---

## 🌍 Real-World Impact

This project contributes toward:

* Precision agriculture
* AI-driven crop monitoring
* Mobile-based plant disease detection
* Sustainable farming practices

Lightweight CNNs like MobileNetV2 can empower farmers with real-time disease diagnostics using smartphones.

---

## 👩‍💻 Author

**Ananya Dua**
