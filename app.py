import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")


# ----------------------------------------------------------------------
@st.cache_data
def load_cancer():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    mean_cols = [c for c in df.columns if c.startswith("mean ")]
    X = df[mean_cols]
    y = df["target"]  # 0 = malignant, 1 = benign
    return X, y


@st.cache_resource
def train_disease(model_name):
    X, y = load_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if model_name == "Logistic Regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif model_name == "SVM":
        model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1-Score": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }
    return model, metrics


def page_disease():
    st.header("🩺 Disease Prediction (Breast Cancer)")
    st.write("Patient ke medical measurements se tumor malignant hai ya benign, ye predict karta hai.")
    model_name = st.selectbox(
        "Model chuno",
        ["Random Forest", "Logistic Regression", "SVM", "Gradient Boosting"],
    )
    model, metrics = train_disease(model_name)
    c = st.columns(5)
    for col, (k, v) in zip(c, metrics.items()):
        col.metric(k, f"{v:.3f}")

    X, _ = load_cancer()
    st.subheader("Patient ki values daalo")
    values = {}
    cols = st.columns(2)
    for i, name in enumerate(X.columns):
        lo, hi, med = float(X[name].min()), float(X[name].max()), float(X[name].median())
        values[name] = cols[i % 2].slider(
            name, lo, hi, med, step=(hi - lo) / 200 or 0.001, format="%.4f"
        )
    if st.button("Predict Disease"):
        row = pd.DataFrame([values])[list(X.columns)]
        p_benign = float(model.predict_proba(row)[0, 1])
        if p_benign >= 0.5:
            st.success(f"Benign (non-cancerous) ✅  (confidence: {p_benign:.1%})")
        else:
            st.error(f"Malignant (cancerous) ⚠️  (confidence: {1 - p_benign:.1%})")
        st.caption("Ye sirf educational project hai, real medical advice nahi.")


page_disease()
