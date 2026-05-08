# 🛡️ Online Fraud Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/)

### 🚀 [Click Here to View Live Demo](https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/)

**A complete end-to-end Machine Learning system that detects fraudulent credit card transactions in real time with 99.99% accuracy**

</div>

---

## 👩‍💻 About The Project

| Detail | Info |
|--------|------|
| **Student** | Sakshi |
| **College** | Gautam Buddha University |
| **Program** | B.Tech CSE (Artificial Intelligence) |
| **Course** | Machine Learning |
| **Year** | 2024-25 |
| **Live App** | [https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/](https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/) |

---

## 🎯 Problem Statement

Credit card fraud causes **billions of dollars** in losses every year. Out of 2,84,807 real transactions only **492 are fraudulent (0.173%)** making detection extremely challenging. Traditional rule-based systems fail to adapt to evolving fraud patterns. This project builds an intelligent ML system that learns from historical data and detects fraud **instantly and accurately**.

---

## 🖥️ Live Demo

<div align="center">

### 🔗 [https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/](https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/)

### Main Dashboard
![Main Dashboard](https://github.com/slashthose/Machine-Learning-Project/blob/main/Outputs/Screenshot%202026-05-08%20093532.png)

### Fraud Detection Result
![Fraud Result](https://github.com/slashthose/Machine-Learning-Project/blob/main/Outputs/Screenshot%202026-05-08%20093641.png)

### Legitimate Transaction Result
![Legit Result](https://github.com/slashthose/Machine-Learning-Project/blob/main/Outputs/Screenshot%202026-05-08%20093612.png)

</div>

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | Kaggle - Credit Card Fraud Detection (ULB) |
| **Total Transactions** | 2,84,807 |
| **Legitimate (Class 0)** | 2,84,315 |
| **Fraudulent (Class 1)** | 492 |
| **Fraud Percentage** | 0.173% |
| **Total Features** | 30 (V1-V28 PCA transformed, Time, Amount) |
| **After SMOTE** | 5,68,630 balanced samples |

> 📥 Download dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🤖 Models and Results

| Model | Accuracy | ROC-AUC | Rank |
|-------|----------|---------|------|
| ⭐ Random Forest | **99.99%** | **1.0000** | 🥇 Best |
| KNN | 99.90% | 0.9997 | 🥈 2nd |
| Neural Network | 99.88% | 0.9995 | 🥉 3rd |
| Decision Tree | 99.83% | 0.9983 | 4th |
| Logistic Regression | 94.84% | 0.9895 | 5th |

---

</div>

---

## ⚙️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.13 |
| **ML Library** | Scikit-learn |
| **Deep Learning** | TensorFlow and Keras |
| **Web App** | Streamlit |
| **Data Handling** | Pandas and NumPy |
| **Visualization** | Matplotlib and Seaborn |
| **Imbalance Fix** | SMOTE (Imbalanced-learn) |
| **Model Saving** | Pickle |
| **IDE** | VS Code |

---

## 📁 Project Structure

```
fraud-detection/
│
├── 📁 data/
│   ├── creditcard.csv              ← dataset (download from Kaggle)
│   └── requirements.txt            ← all libraries
│
├── 📁 outputs/
│   ├── eda_plots.png               ← EDA visualization
│   ├── model_comparison.png        ← model comparison chart
│   ├── best_model_evaluation.png   ← confusion matrix + ROC
│   └── neural_network_training.png ← training graph
│
├── 📁 screenshots/                 ← app screenshots
│   ├── dashboard.png
│   ├── fraud_result.png
│   ├── legit_result.png
│   └── charts.png
│
├── 📄 fraud_detection.py           ← main ML training code
├── 📄 app.py                       ← Streamlit web application
├── 📄 save_model.py                ← saves model.pkl only
├── 📄 model.pkl                    ← saved trained model
├── 📄 run_setup.py                 ← creates project folders
└── 📄 README.md                    ← this file
```

---

## 🚀 How to Run Locally

### Step 1 - Clone Repository
```bash
git clone https://github.com/yourusername/fraud-detection-system
cd fraud-detection-system
```

### Step 2 - Install Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn tensorflow streamlit
```

### Step 3 - Download Dataset
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download `creditcard.csv`
- Place it inside the `data/` folder

### Step 4 - Train Models
```bash
python fraud_detection.py
```
⏳ This takes about 15-20 minutes

### Step 5 - Run Web App
```bash
streamlit run app.py
```

### Step 6 - Open in Browser
```
http://localhost:8501
```

---

## 🌐 Web App Features

### 🔍 Tab 1 - Predict Transaction
- Select from 5 real sample transactions
- Get instant Fraud or Legitimate verdict
- See fraud probability percentage
- View confidence breakdown
- Risk level indicator High Medium Low

### 📊 Tab 2 - Results and Charts
- All 5 model accuracy metrics
- Complete results comparison table
- Accuracy and ROC-AUC bar charts
- Neural network training graph
- Dataset distribution pie chart

### ℹ️ Tab 3 - How It Works
- Complete 6 step pipeline explanation
- Syllabus coverage table
- Full tech stack details

---

## 📚 Syllabus Coverage

| Unit | Topic | Coverage |
|------|-------|----------|
| Unit I | Introduction to ML | Real world fraud detection application |
| Unit II | Supervised Learning | LR, DT, Random Forest, KNN implemented |
| Unit III | Unsupervised Learning | PCA features V1-V28 and SMOTE used |
| Unit IV | Neural Networks | 3 layer MLP with ReLU Sigmoid Backprop |

---

## 🔑 Key Highlights

- ✅ Random Forest achieved **perfect 99.99% accuracy**
- ✅ ROC-AUC of **1.0000** means perfect fraud detection
- ✅ SMOTE balanced dataset from **492 to 5,68,630 samples**
- ✅ Neural Network trained **15 epochs** reached 99.88%
- ✅ **Live web app** deployed on Streamlit Cloud
- ✅ Covers **all 4 units** of ML syllabus in one project
- ✅ **5 models** trained and compared on same dataset

---

## 📜 References

1. [Kaggle Credit Card Fraud Detection Dataset - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Chawla et al. SMOTE Synthetic Minority Oversampling Technique. JAIR 2002
3. Breiman L. Random Forests. Machine Learning Journal 2001
4. [Scikit-learn Documentation](https://scikit-learn.org)
5. [TensorFlow Documentation](https://tensorflow.org)
6. Pedregosa et al. Scikit-learn Machine Learning in Python. JMLR 2011

---

## ⚠️ Note

The `creditcard.csv` dataset is **not included** in this repository due to its large size of 150MB. Please download it from Kaggle using the link above and place it in the `data/` folder before running.

The `model.pkl` file is also **not included** due to its large size of 400-600MB. Run `fraud_detection.py` to generate it.

---

<div align="center">

**🛡️ Online Fraud Detection System**

### 🚀 [View Live Demo](https://machine-learning-project-3wnkaegtpesvzuuerbdrky.streamlit.app/)

## 👨‍💻 Group Members
 
| Name | Roll No |
|------|--------------|
| Sakshi Singh | 245UAI053 |
| Kumkum | 245UAI067 |
| Deepika Suman | 245UAI034 |
| Anmol Verma | 245UAI021 |
 

| Gautam Buddha University | B.Tech CSE (AI) | 2024-25

</div>
