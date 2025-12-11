# Project Status Report – Phase 1: Data Preparation & Initial Training
---

## Phase 1 — Completed Tasks (Data Engineering – Role 1)

### 1. Data Verification & Splitting
All dataset splits were fully verified, and the test-set path issue was resolved.

- Training Set: 70,295 images — 38 classes  
- Validation Set: 14,069 images — 38 classes  
- Test Set: 3,803 images — 38 classes (path corrected)

---

### 2. Preprocessing and Augmentation
A strong preprocessing and augmentation pipeline has been successfully applied:

- Resize all images to 128 × 128  
- Normalize pixel values between 0 and 1  
- Custom CLAHE for contrast enhancement  
- Data augmentation: rotation, zoom, flips, brightness adjustments, shifts

---

### 3. Data Generators
Prepared three main generators:

- train_generator  
- val_generator  
- test_generator

Each generator loads images in batches and provides labels automatically, ensuring smooth integration with the CNN training pipeline.

---

### 4. Class Weights
- Calculated class weights for all 38 classes to handle imbalance.  
- Ensures the model treats all categories fairly.

---

### 5. Visualization
- Generated visualizations (e.g., Pie Chart) to show class distribution.  
- Helps team and supervisor understand dataset balance visually.

---

### 6. Training Readiness
- All preprocessing, augmentation, and generator setups are complete.  
- The training pipeline is fully ready for model development.

---

## Technical Work Summary (Code Notes)

- Code verified and running correctly.  
- Batch shapes and classes are correct:
  - Batch X shape: (32, 128, 128, 3)  
  - Batch y shape: (32, 38)  
- CLAHE, normalization, augmentation, and class weights all implemented.  

### Test Set Note
- If test generator output shows only 1 class, check test folder structure:  
  - Original dataset test folder should have 38 folders (one per class).  
  - Using a single-folder test will only detect 1 class, which is expected.

- Correct path example: