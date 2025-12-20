# Plant Disease Classification 🌱

---
## Overview
This project implements a **Convolutional Neural Network (CNN)** to automatically classify plant leaf images into **38 disease categories**.  
It is designed to assist farmers and agricultural specialists in early detection of plant diseases, improving crop management and yield.  
The model achieves approximately **91% test accuracy** on unseen data.

---
## Dataset
The dataset is sourced from Kaggle: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- **Number of Classes:** 38  
- **Total Images:** 88,167 (Training + Validation + Test)  

---
## Features
- Preprocessing pipeline with resizing, normalization, and CLAHE enhancement  
- Data augmentation for better generalization  
- Class weighting to handle class imbalance  
- CNN architecture: 3 Conv2D layers with MaxPooling, followed by Dense layers with Dropout  

---
## Usage
1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/19mwVk_xIheZnvVrh1t8xtaccOT-0vx4m?authuser=0#scrollTo=zfCR2I65nFm2) to run the project.  
2. Upload a leaf image in the notebook.  
3. The system predicts the disease class of the uploaded image.  

---
## Requirements
- Python 3.x  
- TensorFlow / Keras  
- OpenCV, NumPy, Matplotlib, Scikit-learn  

---
## License
This project is for **educational purposes only**.

---
---
