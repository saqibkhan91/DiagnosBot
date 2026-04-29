# 🩺 DiagnosBot — AI Medical Diagnosis Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen?style=flat)

> An open-source AI-powered disease diagnosis assistant that predicts diseases from symptoms using Machine Learning and NLP.

---

## 📌 Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Model Details](#model-details)
- [Author](#author)

---

## 🎥 Demo

> Select your symptoms → Get instant diagnosis → See precautions & severity

![DiagnosBot Demo](https://raw.githubusercontent.com/saqibkhan91/DiagnosBot/main/assets/demo.png)

---

## ✨ Features

- 🔍 Predicts **41 diseases** from **77 symptoms**
- 🤖 **88% classification accuracy** using Random Forest
- 📊 Compares 3 ML models: Random Forest, SVM, Naive Bayes
- ⚠️ Severity scoring (Mild / Moderate / Severe)
- 💊 Shows disease description and precautions
- 🖥️ Clean interactive Streamlit UI
- 🆓 Completely free and open-source

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | Scikit-learn (Random Forest, SVM, Naive Bayes) |
| NLP | TF-IDF, Feature Engineering |
| UI | Streamlit |
| Data | Pandas, NumPy |
| Model Saving | Joblib |

---

## 📁 Project Structure

```
DiagnosBot/
├── data/
│   ├── dataset.csv              # Disease-symptom dataset
│   ├── symptom_severity.csv     # Symptom severity weights
│   ├── symptom_description.csv  # Disease descriptions
│   └── symptom_precaution.csv   # Recommended precautions
├── model/
│   ├── diagnosbot_model.pkl     # Trained ML model
│   ├── label_encoder.pkl        # Label encoder
│   └── symptoms_list.pkl        # All symptoms list
├── train_model.py               # Model training script
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/saqibkhan91/DiagnosBot.git
cd DiagnosBot
```

### Step 2 — Create Virtual Environment
```bash
# Create venv
python3 -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model
```bash
python3 train_model.py
```
You should see:
```
✅ 164 rows | 41 diseases loaded
✅ 77 unique symptoms found
🏆 Best Model: Random Forest (88.0% accuracy)
💾 Model saved to model/
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 🤖 Model Details

| Model | Accuracy | CV Score |
|---|---|---|
| Random Forest | **88.0%** | 96.9% |
| SVM | 88.0% | 96.9% |
| Naive Bayes | 70.0% | 96.9% |

**Best Model:** Random Forest  
**Features:** 77 binary symptom features  
**Classes:** 41 disease categories  

---

## 👤 Author

**M Saqib Bilal** — Junior Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/saqibkhan91)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/saqibkhan91)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:itssaqibkhan91@gmail.com)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
