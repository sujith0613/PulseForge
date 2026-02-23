# PulseForge  
## Sleep Apnea AHI Estimator  

**YŪGŌ Hackathon 2026 – Pack A Submission** 

**Team:** PulseForge  

**Team Members:**

Sree Krishna S

Sujith M

**Institution:** SSN College of Engineering  

---

## Overview

PulseForge is an end-to-end machine learning system for automated estimation of the **Apnea–Hypopnea Index (AHI)** from overnight ECG recordings.

The system transforms minute-level apnea predictions into a clinically interpretable patient-level AHI score while enforcing strict patient-level validation to eliminate data leakage.

Our pipeline integrates physiological feature engineering, ensemble learning, statistical threshold optimization, and an interactive reporting dashboard.

---



##  Project Structure

```

yugo-submission-sujith0613/
│
├── Dashboard/
│   ├── pulseforge_model.pkl
│   └── streamlit_dashboard.py
│
├── outputs/
├── Apnea.ipynb
├── DISCLAIMER.md
├── README.md
├── requirements.txt
└── .gitignore
````

##  Execution Instructions

## 1️ Clone the Repository

```bash
git clone https://github.com/<your-username>/yugo-submission-sujith0613.git
cd yugo-submission-sujith0613
```

---

## 2 Create a Virtual Environment (Recommended)

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4 Run the Streamlit Application

Since the trained model (`pulseforge_model.pkl`) is already included, no dataset download or retraining is required.

Launch the app:

```bash
streamlit run streamlit_app.py
```

The dashboard will open automatically at:

```
http://localhost:8501
```

---

## 5 What the App Does

* Loads the pre-trained LightGBM model
* Accepts ECG record input
* Extracts physiological features
* Predicts apnea probability per epoch
* Computes AHI
* Classifies severity (Normal / Mild / Moderate / Severe)
* Displays results interactively

---

## 6 Optional: Re-training the Model

If you wish to retrain the model:

Open:

```bash
jupyter notebook Apnea.ipynb
```

(Note: Dataset must be downloaded separately from PhysioNet if retraining.)



## Technical Summary

### Dataset

We trained and evaluated our model using the:

**PhysioNet Apnea-ECG Database**

Goldberger AL et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation 101(23):e215–e220.

- 35 training records  
- 35 test records  
- 60-second annotated epochs  
- Binary apnea labels  

Dataset Source: https://physionet.org/

---

## Feature Engineering (42+ per epoch)

### Statistical Features
- Mean, Standard Deviation, Variance  
- Skewness, Kurtosis  
- RMS, IQR  
- Energy, Zero-crossing rate  
- Peak-to-peak amplitude  

### HRV Features (Pan–Tompkins R-peak detection)
- Mean RR  
- SDNN  
- RMSSD  
- NN50, pNN50  
- Heart Rate  
- CV-RR  
- R-peak amplitude statistics  

### Poincaré Plot Features
- SD1  
- SD2  
- SD2/SD1 ratio  

### Spectral Features (Welch PSD)
- VLF, LF, HF band power  
- LF/HF ratio  
- Spectral entropy  

### RR-Interval Spectral Features
- RR VLF/LF/HF  
- RR LF/HF  
- RR total power  
- HF percentage  

### Temporal Context Features
- ±2 epoch lag features  
- 5-epoch rolling averages  

---

## Model Architecture

- Model: LightGBM Gradient Boosting Classifier  
- Ensemble: 5-fold averaged models  
- Validation: Stratified GroupKFold (patient-level grouping)  
- Threshold Selection: Youden’s J statistic  
- Leakage Prevention: Record ID grouping during cross-validation  

Model citation:

Ke G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems (NeurIPS).

---

## Final Performance (Out-of-Fold)

- **ROC-AUC:** 0.8105  
- **F1-Score:** 0.7412  
- **Optimal Threshold:** ~0.39  

Baseline comparison:
- Baseline AUC ≈ 0.78  
- Baseline F1 ≈ 0.70  

PulseForge demonstrates measurable improvement over the provided baseline.

---

## AHI Computation

AHI = (Predicted Apnea Epochs / Total Epochs) × 60

Severity Classification:

- < 5 → Normal  
- 5–15 → Mild  
- 15–30 → Moderate  
- ≥ 30 → Severe  

---

## Novelty & Innovation

PulseForge advances beyond baseline approaches through:

1. Physiologically grounded HRV feature extraction using Pan–Tompkins R-peak detection.
2. RR-interval spectral decomposition into VLF/LF/HF bands.
3. Poincaré plot-derived nonlinear variability metrics (SD1, SD2).
4. Stratified GroupKFold validation to eliminate patient-level leakage.
5. Short-horizon temporal context modeling via lag features.
6. Ensemble averaging across multiple LightGBM folds for stability.
7. ROC-based statistical threshold optimization rather than arbitrary cutoffs.
8. Fully integrated dashboard + automated clinical-style PDF reporting.

These innovations collectively enhance generalization reliability, physiological interpretability, and deployment readiness.

---

## Intellectual Property / Patent Language Stub

The PulseForge system introduces a structured methodology for transforming ECG-derived physiological signals into a leakage-safe, patient-level AHI estimation pipeline using ensemble machine learning and statistically optimized classification thresholds.

Potentially patentable elements may include:

- A method for physiologically informed apnea detection using combined HRV, spectral, and contextual lag features.
- A leakage-aware validation framework integrated directly into model training for medical time-series classification.
- A system for automated generation of clinically structured AHI reports with embedded model performance transparency.

This repository does not constitute a patent filing but documents the novelty for hackathon evaluation and academic reference.

---

## Dashboard Capabilities

Built using Streamlit:

- Upload PhysioNet Apnea-ECG record files  
- Automated epoch segmentation  
- Feature extraction (42+ per epoch)  
- Ensemble inference  
- Interactive probability timeline  
- Ground-truth validation (if annotations available)  
- Downloadable 1-page PDF clinical report  

---

## Important Notice

This system is developed for research and educational purposes only and does not constitute medical diagnosis.

See DISCLAIMER.md for full legal notice.

---

## Team PulseForge

Sujith M  
Sree Krishna S  

SSN College of Engineering  
YŪGŌ Hackathon 2026





