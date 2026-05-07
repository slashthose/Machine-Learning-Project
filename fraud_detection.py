# ============================================================
#         ONLINE FRAUD DETECTION SYSTEM
#         B.Tech CSE (AI) - Machine Learning Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from imblearn.over_sampling import SMOTE

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("         ONLINE FRAUD DETECTION SYSTEM")
print("=" * 60)
print("\n[1] Loading dataset...")

df = pd.read_csv('data/creditcard.csv')

print(f"    Total Transactions : {len(df)}")
print(f"    Legitimate (0)     : {df['Class'].value_counts()[0]}")
print(f"    Fraudulent (1)     : {df['Class'].value_counts()[1]}")
print(f"    Fraud Percentage   : {round(df['Class'].mean()*100, 3)}%")

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n[2] Generating EDA plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1 - Class Distribution
axes[0].pie(df['Class'].value_counts(),
            labels=['Legitimate', 'Fraud'],
            autopct='%1.2f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90)
axes[0].set_title('Transaction Distribution')

# Plot 2 - Transaction Amount by Class
df.groupby('Class')['Amount'].mean().plot(kind='bar', ax=axes[1],
                                           color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Average Transaction Amount')
axes[1].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[1].set_ylabel('Amount ($)')

# Plot 3 - Amount Distribution
axes[2].hist(df[df['Class']==0]['Amount'], bins=50,
             alpha=0.6, label='Legitimate', color='#2ecc71')
axes[2].hist(df[df['Class']==1]['Amount'], bins=50,
             alpha=0.6, label='Fraud', color='#e74c3c')
axes[2].set_title('Amount Distribution')
axes[2].set_xlabel('Amount')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/eda_plots.png', dpi=150)
plt.show()
print("    EDA plots saved!")

# ============================================================
# STEP 3: PREPROCESSING
# ============================================================
print("\n[3] Preprocessing data...")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale 'Amount' and 'Time' columns
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Handle class imbalance using SMOTE
print("    Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"    After SMOTE - Total samples: {len(X_resampled)}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

print(f"    Training samples : {len(X_train)}")
print(f"    Testing samples  : {len(X_test)}")

# ============================================================
# STEP 4: TRAIN ALL MODELS
# ============================================================
print("\n[4] Training models...")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN"                 : KNeighborsClassifier(n_neighbors=5),
}

results = {}

for name, model in models.items():
    print(f"    Training {name}...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'accuracy' : accuracy_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_prob)
    }
    print(f"Done! Accuracy: {round(results[name]['accuracy']*100, 2)}%")

# ============================================================
# STEP 5: MODEL COMPARISON
# ============================================================
print("\n[5] Comparing models...")

comparison_df = pd.DataFrame({
    'Model'    : list(results.keys()),
    'Accuracy' : [round(v['accuracy']*100, 2) for v in results.values()],
    'ROC-AUC'  : [round(v['roc_auc'], 4) for v in results.values()]
}).sort_values('ROC-AUC', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'], color='#3498db')
axes[0].set_title('Accuracy (%)')
axes[0].set_xlim(80, 100)
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(v + 0.1, i, f'{v}%', va='center')

axes[1].barh(comparison_df['Model'], comparison_df['ROC-AUC'], color='#9b59b6')
axes[1].set_title('ROC-AUC Score')
axes[1].set_xlim(0.8, 1.0)
for i, v in enumerate(comparison_df['ROC-AUC']):
    axes[1].text(v + 0.001, i, f'{v}', va='center')

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150)
plt.show()

# ============================================================
# STEP 6: BEST MODEL - DETAILED EVALUATION
# ============================================================
best_model_name = comparison_df.iloc[0]['Model']
best = results[best_model_name]

print(f"\n[6] Best Model: {best_model_name}")
print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Legitimate', 'Fraud']))

# Confusion Matrix + ROC Curve side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_model_name}', fontsize=16, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best['y_prob'])
axes[1].plot(fpr, tpr, color='#e74c3c',
             label=f'ROC Curve (AUC = {round(best["roc_auc"], 4)})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/best_model_evaluation.png', dpi=150)
plt.show()

# ============================================================
# STEP 7: NEURAL NETWORK (Unit IV Coverage)
# ============================================================
print("\n[7] Training Neural Network...")

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    nn_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    history = nn_model.fit(X_train, y_train,
                           epochs=15,
                           batch_size=64,
                           validation_split=0.1,
                           verbose=1)

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Neural Network Training', fontsize=16, fontweight='bold')

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/neural_network_training.png', dpi=150)
    plt.show()

    nn_loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
    print(f"    Neural Network Accuracy: {round(nn_acc*100, 2)}%")

except ImportError:
    print("    TensorFlow not installed, skipping Neural Network.")

# ============================================================
# STEP 8: SAVE BEST MODEL
# ============================================================
print("\n[8] Saving best model...")

with open('model.pkl', 'wb') as f:
    pickle.dump(results[best_model_name]['model'], f)

print(f"    Model saved as 'model.pkl'")

# ============================================================
# STEP 9: PREDICT ON NEW TRANSACTION
# ============================================================
print("\n[9] Testing prediction on a sample transaction...")

sample = X_test.iloc[0].values.reshape(1, -1)
prediction = results[best_model_name]['model'].predict(sample)
probability = results[best_model_name]['model'].predict_proba(sample)[0][1]

print(f"    Prediction  : {'🚨 FRAUD' if prediction[0]==1 else '✅ LEGITIMATE'}")
print(f"    Fraud Prob  : {round(probability*100, 2)}%")

print("\n" + "="*60)
print("   PROJECT COMPLETE!")
print("   All plots saved in 'outputs/' folder")
print("="*60)