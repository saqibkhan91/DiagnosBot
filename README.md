# 🩺 DiagnosBot — AI Medical Diagnosis Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Live](https://img.shields.io/badge/Live-Demo-blue?style=flat&logo=huggingface)

> An open-source AI-powered disease diagnosis assistant that predicts diseases from symptoms using Machine Learning and NLP. Built for global health equity — especially for low-resource healthcare settings.

---

## 🌐 Live Demo
👉 **https://huggingface.co/spaces/saqibkhan91/DiagnosBot**

> No installation needed — just open the link and start diagnosing!

---

## 📌 Table of Contents
- [Version History](#version-history)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Model Details](#model-details)
- [Roadmap](#roadmap)
- [Author](#author)

---

## 🔄 Version History

### ✅ v1.0 — Foundation (February 2026)
> First working version of DiagnosBot

**What was built:**
- 🗃️ Custom disease-symptom dataset — **164 rows, 41 diseases, 77 symptoms**
- 🤖 Trained and compared **3 ML models**: Random Forest, SVM, Naive Bayes
- 🏆 Best model: **Random Forest — 88% accuracy**
- ⚠️ Severity scoring system (Mild / Moderate / Severe)
- 💊 Disease description + precautions display
- 🖥️ Basic Streamlit UI with symptom dropdown

**Limitations of v1.0:**
- Small dataset (164 rows only)
- Only 4 symptoms per disease
- No confidence score
- No free text input
- Basic UI design

---

### ✅ v2.0 — Accuracy Upgrade (April 2026)
> Major upgrade — 30x larger dataset, 100% accuracy, live deployment

**What was upgraded:**
- 🗃️ Replaced dataset with **Kaggle full dataset — 4,920 rows, 41 diseases, 132 symptoms**
- 🤖 Trained **3 models + GridSearchCV hyperparameter tuning**
- 🏆 Achieved **100% classification accuracy**
- 🎯 **Top 3 disease predictions** with confidence percentages
- 📊 **Confidence gauge chart** showing diagnosis certainty
- 🔍 **Symptom search bar** — filter from 132 symptoms instantly
- 📈 **Feature importance chart** — top 15 most diagnostic symptoms
- 🗂️ **2-tab UI**: Diagnose tab + Symptom Insights tab
- 🌐 **Deployed live on Hugging Face Spaces** — publicly accessible worldwide
- 🔄 **Self-training model** — trains on startup, no pkl files needed

**Improvements over v1.0:**

| Metric | v1.0 | v2.0 |
|---|---|---|
| Dataset rows | 164 | **4,920** (+30x) |
| Symptoms | 77 | **132** (+71%) |
| Accuracy | 88% | **100%** |
| Predictions shown | 1 | **Top 3** |
| Confidence gauge | ❌ | ✅ |
| Symptom search | ❌ | ✅ |
| Feature importance | ❌ | ✅ |
| Live deployment | ❌ | ✅ |
| Tabs UI | ❌ | ✅ |

---

### 🔜 v3.0 — Smart Features (Coming Soon)
- Free text symptom input using NLP
- Arabic language support (UAE market)
- Patient history tracking
- Doctor type recommendation
- PDF diagnosis report export
- Mobile-friendly responsive UI

### 🔜 v4.0 — LLM Integration (Planned)
- GPT/Gemini API integration
- Natural conversation diagnosis
- Drug interaction warnings
- User accounts and history
- Mobile app (React Native)

---

## ✨ Current Features (v2.0)

- 🔍 Predicts **41 diseases** from **132 symptoms**
- 🤖 **100% classification accuracy** using Random Forest
- 🎯 Shows **Top 3 possible diagnoses** with confidence %
- 📊 **Confidence gauge chart** for visual clarity
- 🔍 **Real-time symptom search** from 132 symptoms
- ⚠️ **Severity scoring** (Mild / Moderate / Severe)
- 💊 Disease description and precautions
- 📈 **Feature importance chart** showing key symptoms
- 🌐 **Live on Hugging Face** — no installation needed
- 🆓 Completely free and open-source

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | Scikit-learn (Random Forest, SVM, Naive Bayes) |
| Tuning | GridSearchCV |
| UI | Streamlit |
| Charts | Plotly |
| Data | Pandas, NumPy |
| Deployment | Hugging Face Spaces, Docker |

---

## 📁 Project Structure

```
DiagnosBot/
├── data/
│   ├── symptom_severity.csv      # Symptom severity weights
│   ├── symptom_description.csv   # Disease descriptions
│   └── symptom_precaution.csv    # Recommended precautions
├── data_v2/
│   ├── Training.csv              # v2.0 training dataset (4920 rows)
│   └── Testing.csv               # v2.0 testing dataset (41 rows)
├── model/                        # v1.0 saved models
│   ├── diagnosbot_model.pkl
│   ├── label_encoder.pkl
│   └── symptoms_list.pkl
├── app.py                        # v1.0 Streamlit app
├── app_v2.py                     # v2.0 Streamlit app (latest)
├── train_model.py                # v1.0 training script
├── train_model_v2.py             # v2.0 training script
├── requirements.txt              # Python dependencies
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

### Step 4 — Train the Model (v2.0)
```bash
python3 train_model_v2.py
```

Expected output:
```
✅ Train: (4920, 132) | Test: (41, 132)
✅ Diseases: 41 | Symptoms: 132
Random Forest      → Accuracy: 100.0%
SVM                → Accuracy: 100.0%
Gradient Boosting  → Accuracy: 100.0%
🏆 FINAL MODEL: Random Forest → 100.0%
💾 Model saved to model_v2/
```

---

## 🚀 How to Run

### Run v2.0 (Recommended)
```bash
streamlit run app_v2.py
```

### Run v1.0 (Legacy)
```bash
streamlit run app.py
```

Open browser at: **http://localhost:8501**

### Or use Live Demo (No installation!)
👉 **https://huggingface.co/spaces/saqibkhan91/DiagnosBot**

---

## 🤖 Model Details

### v2.0 Models Comparison

| Model | Accuracy | CV Score |
|---|---|---|
| Random Forest | **100.0%** ✅ | 100.0% |
| SVM | 100.0% | 100.0% |
| Gradient Boosting | 100.0% | 100.0% |

### v1.0 Models Comparison

| Model | Accuracy | CV Score |
|---|---|---|
| Random Forest | **88.0%** ✅ | 96.9% |
| SVM | 88.0% | 96.9% |
| Naive Bayes | 70.0% | 96.9% |

---

## 🗺️ Roadmap

- [x] v1.0 — Basic diagnosis (88% accuracy)
- [x] v2.0 — Full dataset, 100% accuracy, live deployment
- [ ] v3.0 — Free text input, Arabic support, PDF report
- [ ] v4.0 — LLM integration, mobile app
- [ ] v5.0 — Hospital partnerships, real patient data

---

## 👤 Author

**M Saqib Bilal** — Junior Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/saqibkhan91)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/saqibkhan91)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:itssaqibkhan91@gmail.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Live_Demo-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/saqibkhan91/DiagnosBot)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
