🛡️ Online Fraud Detection System

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions in real time using multiple ML algorithms and a live Streamlit web application.

👩‍💻 About the Project

Student: Sakshi
College: Gautam Buddha University
Program: B.Tech CSE (Artificial Intelligence)
Course: Machine Learning
Academic Year: 2024–25

This project was developed as part of the Machine Learning curriculum to demonstrate how AI can solve real-world financial security problems. The system analyzes transaction patterns and predicts whether a transaction is legitimate or fraudulent.

🎯 Problem Statement

Credit card fraud has become a major issue in the digital world, causing huge financial losses every year. One of the biggest challenges is that fraudulent transactions are extremely rare compared to normal ones.

In this dataset:

Total transactions: 284,807
Fraud cases: 492 only
Fraud percentage: 0.173%

Because of this imbalance, detecting fraud accurately is very difficult.
This project solves the problem using Machine Learning techniques and SMOTE balancing, achieving up to 99.99% accuracy.

📊 Dataset Information

Dataset Source: Kaggle – Credit Card Fraud Detection (ULB)

Dataset Details
Transactions: 284,807
Fraud Cases: 492
Features: 30
V1–V28 → PCA transformed features
Time
Amount

The dataset contains real anonymized credit card transaction data collected by European cardholders.

🤖 Models Used and Performance
Model	Accuracy	ROC-AUC Score
Random Forest ⭐	99.99%	1.0000
KNN	99.90%	0.9997
Neural Network	99.88%	0.9995
Decision Tree	99.83%	0.9983
Logistic Regression	94.84%	0.9895
Best Performing Model

The Random Forest Classifier performed the best with:

99.99% Accuracy
1.0000 ROC-AUC Score

This means the model was able to identify almost every fraudulent transaction correctly.

⚙️ Technologies and Tools Used
Programming Language: Python 3.13
Machine Learning: Scikit-learn
Deep Learning: TensorFlow & Keras
Web Application: Streamlit
Data Processing: Pandas & NumPy
Visualization: Matplotlib & Seaborn
Imbalance Handling: SMOTE (Imbalanced-learn)

📁 Project Structure
fraud-detection/
├── data/
│   ├── creditcard.csv
│   └── requirements.txt
├── outputs/
│   ├── eda_plots.png
│   ├── model_comparison.png
│   ├── best_model_evaluation.png
│   └── neural_network_training.png
├── app.py
├── fraud_detection.py
├── model.pkl
├── run_setup.py
└── README.md


📈 Key Highlights
Random Forest achieved nearly perfect fraud detection performance
SMOTE balanced the dataset successfully
Neural Network reached 99.88% accuracy after 15 epochs
A live Streamlit web app was developed for real-time predictions
Multiple ML models were trained and compared
The project covers all major Machine Learning syllabus concepts



📚 Syllabus Coverage
Unit I
Real-world Machine Learning application using fraud detection

Unit II
Implementation of:
Logistic Regression
Decision Tree
Random Forest
KNN

Unit III
Use of:
PCA-transformed features
SMOTE balancing technique

Unit IV
Deep Learning concepts:
Multilayer Perceptron (MLP)
ReLU Activation
Sigmoid Activation
Backpropagation

📜 References
Kaggle Credit Card Fraud Detection Dataset (ULB)
Chawla et al., “SMOTE Technique,” JAIR, 2002
Breiman L., “Random Forests,” Machine Learning Journal, 2001
Scikit-learn Documentation
TensorFlow Documentation
