# -*- coding: utf-8 -*-
# ============================================================
#   ONLINE FRAUD DETECTION SYSTEM - STREAMLIT UI
#   Student : Sakshi
#   College : Gautam Buddha University
#   Run     : streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | Sakshi | GBU",
    page_icon="🛡️",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🛡️ Online Fraud Detection System")
st.markdown("**Sakshi &nbsp;|&nbsp; Gautam Buddha University &nbsp;|&nbsp; B.Tech CSE (AI) &nbsp;|&nbsp; Machine Learning Project 2024-25**")
st.markdown("---")

# ── Load Model ───────────────────────────────────────────────
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ model.pkl not found. Running in DEMO mode.")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/b/b4/Gautam_Buddha_University_logo.png", width=120)
st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
**Student:** Sakshi  
**College:** Gautam Buddha University  
**Program:** B.Tech CSE (AI)  
**Year:** 2024-25  

---

**Dataset:** Credit Card Fraud Detection  
**Source:** Kaggle (ULB)  
**Total Records:** 2,84,807  
**Fraud Cases:** 492 (0.173%)  
**After SMOTE:** 5,68,630  

---

**Best Model:** Random Forest  
**Best Accuracy:** 99.99%  
**Best ROC-AUC:** 1.0000  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Results:**")
df_perf = pd.DataFrame({
    "Model":     ["Random Forest", "KNN", "Decision Tree", "Logistic Reg", "Neural Net"],
    "Accuracy":  ["99.99%", "99.90%", "99.83%", "94.84%", "99.88%"],
    "ROC-AUC":   ["1.0000", "0.9997", "0.9983", "0.9895", "0.9995"],
})
st.sidebar.dataframe(df_perf, hide_index=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Transaction", "📊 Results & Charts", "ℹ️ How It Works"])

# ════════════════════════════════════════════════════════════
# TAB 1 - PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Enter Transaction Details")
    st.info("Fill in the transaction features below and click **Analyze** to detect fraud.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Transaction Info**")
        amount = st.number_input("💰 Transaction Amount (₹)", min_value=0.01, max_value=50000000.0, value=100.0)
        time   = st.number_input("⏱️ Time (seconds elapsed)", min_value=0, max_value=200000, value=50000)

        st.markdown("**PCA Features V1 - V10**")
        v = {}
        for i in range(1, 11):
            v[i] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"va{i}")

    with col2:
        st.markdown("**PCA Features V11 - V20**")
        for i in range(11, 21):
            v[i] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"vb{i}")

        st.markdown("**PCA Features V21 - V28**")
        for i in range(21, 29):
            v[i] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"vc{i}")

    st.markdown("---")
    analyze = st.button("🔍 ANALYZE TRANSACTION", use_container_width=True)

    if analyze:
        feature_vals = [v[i] for i in range(1, 29)] + [time, amount]
        input_array  = np.array(feature_vals).reshape(1, -1)

        if model is not None:
            try:
                prediction = model.predict(input_array)[0]
                proba      = model.predict_proba(input_array)[0]
                fraud_prob = round(proba[1] * 100, 2)
                legit_prob = round(proba[0] * 100, 2)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                prediction = 0
                fraud_prob = 5.0
                legit_prob = 95.0
        else:
            fraud_prob = round(np.random.uniform(5, 95), 2)
            legit_prob = round(100 - fraud_prob, 2)
            prediction = 1 if fraud_prob > 50 else 0

        st.markdown("### 🎯 Analysis Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Verdict",      "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE")
        r2.metric("Fraud Risk",   f"{fraud_prob}%")
        r3.metric("Legit Score",  f"{legit_prob}%")
        r4.metric("Amount",       f"₹{amount:,.2f}")

        if prediction == 1:
            st.error(f"""
            🚨 **FRAUDULENT TRANSACTION DETECTED!**

            This transaction has been flagged as potentially fraudulent.  
            Fraud Probability: **{fraud_prob}%**  
            Recommended Action: **Block transaction and alert the customer immediately.**
            """)
        else:
            st.success(f"""
            ✅ **LEGITIMATE TRANSACTION**

            This transaction appears to be genuine.  
            Legitimacy Score: **{legit_prob}%**  
            Recommended Action: **Approve and process the transaction.**
            """)

        st.markdown("**Confidence Breakdown:**")
        st.progress(int(fraud_prob), text=f"Fraud Risk: {fraud_prob}%")
        st.progress(int(legit_prob), text=f"Legitimate: {legit_prob}%")

        risk = "🔴 HIGH RISK" if fraud_prob > 70 else ("🟡 MEDIUM RISK" if fraud_prob > 30 else "🟢 LOW RISK")
        st.markdown(f"**Overall Risk Level: {risk}**")

# ════════════════════════════════════════════════════════════
# TAB 2 - RESULTS
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Actual Model Results")
    st.success("These are the REAL results from training on the Credit Card Fraud dataset.")

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Random Forest",  "99.99%", "Best Model ⭐")
    c2.metric("KNN",            "99.90%", "2nd Best")
    c3.metric("Neural Network", "99.88%", "Deep Learning")
    c4.metric("Decision Tree",  "99.83%", "Interpretable")
    c5.metric("Logistic Reg.",  "94.84%", "Baseline")

    st.markdown("---")

    # Full results table
    st.markdown("**Complete Results Table (Your Actual Output):**")
    results_df = pd.DataFrame({
        "Model":          ["Random Forest ⭐", "KNN", "Neural Network", "Decision Tree", "Logistic Regression"],
        "Accuracy":       ["99.99%", "99.90%", "99.88%", "99.83%", "94.84%"],
        "Precision":      ["1.00", "1.00", "1.00", "1.00", "0.95"],
        "Recall":         ["1.00", "1.00", "1.00", "1.00", "0.95"],
        "F1-Score":       ["1.00", "1.00", "1.00", "1.00", "0.95"],
        "ROC-AUC":        ["1.0000", "0.9997", "0.9995", "0.9983", "0.9895"],
    })
    st.dataframe(results_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Dataset info
    st.markdown("**Dataset Statistics (Your Actual Output):**")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Transactions", "2,84,807")
    d2.metric("Legitimate",         "2,84,315")
    d3.metric("Fraudulent",         "492")
    d4.metric("Fraud %",            "0.173%")
    d5.metric("After SMOTE",        "5,68,630")

    st.markdown("---")

    # Charts
    st.markdown("**Accuracy Comparison Chart:**")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    models_list = ["Random\nForest", "KNN", "Neural\nNet", "Decision\nTree", "Logistic\nReg"]
    accuracy    = [99.99, 99.90, 99.88, 99.83, 94.84]
    roc         = [1.0000, 0.9997, 0.9995, 0.9983, 0.9895]
    colors      = ["#E74C3C", "#3498DB", "#9B59B6", "#2ECC71", "#F39C12"]

    bars = axes[0].bar(models_list, accuracy, color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("Accuracy (%) - Sakshi's Results", fontweight="bold", fontsize=12)
    axes[0].set_ylim(92, 100.5)
    axes[0].set_ylabel("Accuracy (%)")
    for bar, val in zip(bars, accuracy):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val}%", ha="center", fontsize=9, fontweight="bold")

    bars2 = axes[1].bar(models_list, roc, color=colors, edgecolor="white", width=0.6)
    axes[1].set_title("ROC-AUC Score - Sakshi's Results", fontweight="bold", fontsize=12)
    axes[1].set_ylim(0.985, 1.002)
    axes[1].set_ylabel("ROC-AUC")
    for bar, val in zip(bars2, roc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                     f"{val}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Pie chart
    st.markdown("**Dataset Class Distribution:**")
    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.pie([284315, 492],
           labels=["Legitimate (99.827%)", "Fraud (0.173%)"],
           colors=["#2ECC71", "#E74C3C"],
           autopct="%1.3f%%", startangle=90,
           explode=(0, 0.1))
    ax.set_title("Credit Card Transaction Distribution", fontweight="bold")
    st.pyplot(fig2)
    plt.close()

    # Neural network training
    st.markdown("---")
    st.markdown("**Neural Network Training Progress (15 Epochs):**")
    epochs   = list(range(1, 16))
    train_acc = [0.9822, 0.9964, 0.9974, 0.9979, 0.9982, 0.9983,
                 0.9985, 0.9986, 0.9987, 0.9987, 0.9988, 0.9988,
                 0.9988, 0.9989, 0.9990]
    val_acc  = [0.9970, 0.9986, 0.9993, 0.9994, 0.9994, 0.9995,
                0.9990, 0.9995, 0.9995, 0.9996, 0.9995, 0.9995,
                0.9995, 0.9995, 0.9988]

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(epochs, train_acc, marker='o', label='Train Accuracy', color='#3498DB', linewidth=2)
    ax3.plot(epochs, val_acc,   marker='s', label='Val Accuracy',   color='#E74C3C', linewidth=2)
    ax3.set_title("Neural Network Training - Sakshi's Results", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    ax3.set_ylim(0.97, 1.001)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 3 - HOW IT WORKS
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ How The System Works")

    st.markdown("""
    ### 🔄 Project Pipeline

    **Step 1 - Data Loading**  
    2,84,807 real credit card transactions loaded from Kaggle dataset.
    492 are fraudulent (only 0.173%).

    **Step 2 - Preprocessing**  
    StandardScaler applied to normalize Amount and Time features.
    SMOTE applied to balance dataset from 492 fraud cases to 2,84,315 — total 5,68,630 samples.

    **Step 3 - Model Training**  
    4 ML models + 1 Neural Network trained on 4,54,904 training samples.

    **Step 4 - Evaluation**  
    Tested on 1,13,726 samples. Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC.

    **Step 5 - Best Model Selected**  
    Random Forest achieved perfect 99.99% accuracy and 1.0000 ROC-AUC.

    **Step 6 - Real-time Prediction**  
    New transactions analyzed instantly and flagged as Fraud or Legitimate.
    """)

    st.markdown("---")
    st.markdown("### 📚 Syllabus Coverage")
    syllabus_df = pd.DataFrame({
        "Unit":    ["Unit I", "Unit II", "Unit III", "Unit IV"],
        "Topic":   ["Intro to ML", "Supervised Learning", "Unsupervised Learning", "Neural Networks"],
        "Covered In This Project": [
            "Real-world fraud detection application",
            "Logistic Regression, Decision Tree, Random Forest, KNN",
            "PCA features V1-V28 + SMOTE for balancing",
            "3-layer MLP: ReLU + Sigmoid + Backpropagation"
        ],
        "Status": ["✅ Done", "✅ Done", "✅ Done", "✅ Done"]
    })
    st.dataframe(syllabus_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack Used")
    t1, t2, t3, t4 = st.columns(4)
    t1.info("**Language**\nPython 3.13")
    t2.info("**ML Library**\nScikit-learn")
    t3.info("**Deep Learning**\nTensorFlow/Keras")
    t4.info("**UI Framework**\nStreamlit")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*🛡️ Online Fraud Detection System &nbsp;|&nbsp; Sakshi &nbsp;|&nbsp; "
    "Gautam Buddha University &nbsp;|&nbsp; B.Tech CSE (AI) &nbsp;|&nbsp; 2024-25*"
)
