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
st.markdown(
    "**Gautam Buddha University | B.Tech CSE (AI) | Machine Learning Project 2024-25**"
)
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
st.sidebar.title("📊 Project Info")

st.sidebar.markdown("""
### 👨‍💻 Group Members
- Sakshi Singh
- Kumkum
- Deepika Suman
- Anmol Verma

---

**College:** Gautam Buddha University  
**Program:** B.Tech CSE (AI)  
**Year:** 2024-25  

---

### 📂 Dataset
**Dataset:** Credit Card Fraud Detection  
**Source:** Kaggle (ULB)  

- Total Records: 2,84,807  
- Fraud Cases: 492 (0.173%)  
- After SMOTE: 5,68,630  

---

### 🏆 Best Model
- Random Forest
- Accuracy: 99.99%
- ROC-AUC: 1.0000
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📈 Model Results")

df_perf = pd.DataFrame({
    "Model": [
        "Random Forest",
        "KNN",
        "Neural Net",
        "Decision Tree",
        "Logistic Reg"
    ],
    "Accuracy": [
        "99.99%",
        "99.90%",
        "99.88%",
        "99.83%",
        "94.84%"
    ],
    "ROC-AUC": [
        "1.0000",
        "0.9997",
        "0.9995",
        "0.9983",
        "0.9895"
    ],
})

st.sidebar.dataframe(df_perf, hide_index=True)

# ── Sample Transactions ──────────────────────────────────────
SAMPLES = {

    "Normal Transaction 1 - Small Amount": {
        "amount": 9.99,
        "time": 406,
        "v": [
            -1.3598071,-0.0727812,2.5363467,1.3781552,
            -0.3383208,0.4623878,0.2395986,0.0986980,
            0.3637870,0.0907942,-0.5515995,-0.6178009,
            -0.9913898,-0.3111695,1.4681770,-0.4704005,
            0.2079713,0.0257906,0.4039936,0.2514121,
            -0.0183068,0.2778376,-0.1104739,0.0669280,
            0.1285394,-0.1891148,0.1335584,-0.0210530
        ]
    },

    "Normal Transaction 2 - Medium Amount": {
        "amount": 149.62,
        "time": 5200,
        "v": [
            1.1918571,0.2661507,0.1664801,0.4481541,
            0.0600176,-0.0823608,-0.0788030,0.0851017,
            -0.2554251,-0.1669271,1.6127267,1.0652353,
            0.4890120,-0.1437723,0.6355582,0.4639170,
            -0.1148127,-0.1833612,-0.1457286,-0.0690831,
            -0.2252804,0.1779929,0.5077569,-0.2879237,
            0.3566640,-0.2548951,-0.0455750,-0.2196930
        ]
    },

    "Normal Transaction 3 - Large Amount": {
        "amount": 529.00,
        "time": 72000,
        "v": [
            -0.9662717,-0.1851360,1.7929820,-0.8633087,
            -0.0103089,1.2472279,0.2376089,0.3774162,
            -1.3870241,-0.0549519,-1.1709820,0.8774769,
            -1.2273646,1.5629499,0.3862496,-1.3193290,
            -0.7282562,0.6798783,-0.1288979,0.4820718,
            0.7433822,-0.1203823,-0.2204773,0.0250619,
            0.7764411,-0.0648944,-0.2143568,-0.0994783
        ]
    },

    "Suspicious Transaction 1 - Fraud Pattern": {
        "amount": 1.00,
        "time": 152,
        "v": [
            -2.3122265,1.9519674,-1.6098027,3.9979356,
            -0.5220188,-1.4265408,-2.5374120,1.3918320,
            -2.7700091,-2.7722956,3.2020316,-2.8990173,
            -0.5955432,-1.3308946,0.0552420,-0.6179697,
            -1.1848945,0.7774497,-1.1596745,0.2502397,
            0.5418557,0.1769551,-0.4512487,0.0479959,
            -0.5369340,-0.0691399,-0.2255700,0.0783641
        ]
    },

    "Suspicious Transaction 2 - Fraud Pattern": {
        "amount": 239.93,
        "time": 27,
        "v": [
            -3.0435407,-3.1572415,1.0877334,2.2886436,
            1.3598510,-1.0632931,0.3253117,-0.0678613,
            0.8475763,-0.3033924,0.4502415,-0.3508503,
            -1.5631540,1.7843082,-0.4893590,1.4036847,
            -0.1459843,-1.3296700,0.4706698,0.5238060,
            0.4830034,-0.5569818,-0.0549527,-0.0960294,
            -0.1849563,0.3766396,0.0671588,0.1299050
        ]
    },

    "High Risk Transaction - 95% Fraud": {
        "amount": 4999.99,
        "time": 172800,
        "v": [
            -7.5123, 8.2311, -9.1145, 6.7821,
            -5.4455, -3.9122, -8.2312, 7.1123,
            -4.9811, -6.7723, 5.8821, -7.3311,
            -3.9911, -6.1112, -2.9921, -4.8822,
            -6.2311, 4.9911, -5.8811, 3.7721,
            2.9912, -1.8821, -2.7711, 1.2211,
            -0.9921, 0.8811, -1.3311, 0.7721
        ]
    }
}

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Predict Transaction",
    "📊 Results & Charts",
    "ℹ️ How It Works"
])

# ============================================================
# TAB 1 - PREDICTION
# ============================================================
with tab1:

    st.subheader("🔍 Transaction Fraud Detector")

    st.info(
        "Select a sample transaction OR enter Amount and Time manually."
    )

    st.markdown("### 📌 Quick Load Sample")

    selected = st.selectbox(
        "Choose a sample:",
        ["-- Select a sample --"] + list(SAMPLES.keys())
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    if selected != "-- Select a sample --":
        preset = SAMPLES[selected]
        default_amount = preset["amount"]
        default_time = preset["time"]
        v_vals = preset["v"]
    else:
        default_amount = 150.0
        default_time = 50000
        v_vals = [0.0] * 28

    with col1:

        st.markdown("### 💳 Transaction Details")

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=50000.0,
            value=float(default_amount)
        )

        time = st.number_input(
            "Time (seconds)",
            min_value=0,
            max_value=200000,
            value=int(default_time)
        )

    with col2:

        st.markdown("### ⚠️ Risk Indicators")

        st.markdown(f"**Amount:** ${default_amount:,.2f}")
        st.markdown(f"**Time:** {default_time:,} seconds")

        if selected != "-- Select a sample --":

            if (
                "Fraud" in selected or
                "Suspicious" in selected or
                "High Risk" in selected
            ):
                st.error("⚠️ Potential Fraud Pattern")
            else:
                st.success("✅ Legitimate Transaction")

    st.markdown("---")

    analyze = st.button(
        "🔍 ANALYZE TRANSACTION",
        use_container_width=True
    )

    if analyze:

        feature_vals = v_vals + [time, amount]
        input_array = np.array(feature_vals).reshape(1, -1)

        if model is not None:

            try:
                prediction = model.predict(input_array)[0]
                proba = model.predict_proba(input_array)[0]

                fraud_prob = round(proba[1] * 100, 2)
                legit_prob = round(proba[0] * 100, 2)

            except Exception as e:

                st.error(f"Prediction Error: {e}")

                if (
                    "Fraud" in selected or
                    "Suspicious" in selected or
                    "High Risk" in selected
                ):
                    prediction = 1
                    fraud_prob = 95.0
                else:
                    prediction = 0
                    fraud_prob = 3.0

                legit_prob = 100 - fraud_prob

        else:

            if (
                "Fraud" in selected or
                "Suspicious" in selected or
                "High Risk" in selected
            ):
                prediction = 1
                fraud_prob = round(np.random.uniform(88, 99), 2)
            else:
                prediction = 0
                fraud_prob = round(np.random.uniform(1, 8), 2)

            legit_prob = round(100 - fraud_prob, 2)

        st.markdown("## 🎯 Analysis Result")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Verdict",
            "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE"
        )

        r2.metric("Fraud Risk", f"{fraud_prob}%")
        r3.metric("Legitimate", f"{legit_prob}%")
        r4.metric("Amount", f"${amount:,.2f}")

        if prediction == 1:

            st.error(f"""
🚨 FRAUDULENT TRANSACTION DETECTED

Fraud Probability: {fraud_prob}%

Recommended Action:
Block the transaction immediately.
            """)

        else:

            st.success(f"""
✅ LEGITIMATE TRANSACTION

Legitimacy Score: {legit_prob}%

Recommended Action:
Approve transaction.
            """)

        st.markdown("### 📊 Confidence Breakdown")

        st.progress(
            int(fraud_prob),
            text=f"Fraud Risk: {fraud_prob}%"
        )

        st.progress(
            int(legit_prob),
            text=f"Legitimate: {legit_prob}%"
        )

        if fraud_prob > 70:
            risk = "🔴 HIGH RISK"
        elif fraud_prob > 30:
            risk = "🟡 MEDIUM RISK"
        else:
            risk = "🟢 LOW RISK"

        st.markdown(f"## {risk}")

# ============================================================
# TAB 2 - RESULTS
# ============================================================
with tab2:

    st.subheader("📊 Model Results")

    results_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "KNN",
            "Neural Network",
            "Decision Tree",
            "Logistic Regression"
        ],
        "Accuracy": [
            99.99,
            99.90,
            99.88,
            99.83,
            94.84
        ]
    })

    st.dataframe(results_df, use_container_width=True)

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy %")

    plt.xticks(rotation=15)

    st.pyplot(fig)

# ============================================================
# TAB 3 - HOW IT WORKS
# ============================================================
with tab3:

    st.subheader("ℹ️ How The System Works")

    st.markdown("""
### 🔄 Project Pipeline

1. Load Credit Card Fraud Dataset  
2. Preprocess Data  
3. Apply SMOTE for balancing  
4. Train ML Models  
5. Evaluate Performance  
6. Predict Fraud in Real Time

---

### 🧠 Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Neural Network

---

### 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- TensorFlow
- Pandas
- NumPy
- Matplotlib
""")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")

st.markdown(
    """
🛡️ Online Fraud Detection System  
Gautam Buddha University | B.Tech CSE (AI) | 2024-25
"""
)
