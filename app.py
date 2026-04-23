import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f3c73;
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f3c73, #2d6ae0);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f4ff;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Feature Definitions ────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "Transaction_Amount":            "Amount of the transaction (₹)",
    "Transaction_Type":              "Type of transaction (encoded)",
    "Account_Balance":               "Account balance at time of transaction (₹)",
    "Device_Type":                   "Device used (encoded)",
    "Location":                      "Transaction location (encoded)",
    "Merchant_Category":             "Category of merchant (encoded)",
    "IP_Address_Flag":               "Flagged IP address (0/1)",
    "Previous_Fraudulent_Activity":  "Prior fraud history (0/1)",
    "Daily_Transaction_Count":       "Number of transactions today",
    "Avg_Transaction_Amount_7d":     "7-day average transaction amount (₹)",
    "Failed_Transaction_Count_7d":   "Failed transactions in last 7 days",
    "Card_Type":                     "Card type (encoded)",
    "Card_Age":                      "Age of card (months)",
    "Transaction_Distance":          "Distance from usual location (km)",
    "Authentication_Method":         "Auth method used (encoded)",
    "Risk_Score":                    "Pre-computed risk score",
    "Is_Weekend":                    "Transaction on weekend (0/1)",
    "Hour":                          "Hour of transaction (0–23)",
    "DayOfWeek":                     "Day of week (0=Mon)",
    "Month":                         "Month of transaction (1–12)",
}

TARGET = "Fraud_Label"
FEATURES = list(FEATURE_DESCRIPTIONS.keys())

# ─── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("fraud_detection_cleaned.csv")
    return df

# ─── Train Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df, selected_features, model_name, test_size, random_state):
    X = df[selected_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=random_state),
    }
    clf = models[model_name]
    clf.fit(X_train_s, y_train)

    y_pred      = clf.predict(X_test_s)
    y_prob      = clf.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "cm":        confusion_matrix(y_test, y_pred),
        "fpr":       roc_curve(y_test, y_prob)[0],
        "tpr":       roc_curve(y_test, y_prob)[1],
        "precision": precision_recall_curve(y_test, y_prob)[0],
        "recall":    precision_recall_curve(y_test, y_prob)[1],
        "y_test":    y_test,
        "y_prob":    y_prob,
    }

    # Feature importances
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
    else:
        fi = pd.Series(np.abs(clf.coef_[0]), index=selected_features).sort_values(ascending=False)

    return clf, scaler, metrics, fi

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/fraud.png", width=60)
    st.title("⚙️ Controls")
    st.divider()

    # Upload
    uploaded = st.file_uploader("📂 Upload CSV (optional)", type=["csv"])

    st.subheader("🎯 Target Variable")
    st.info(f"**{TARGET}** — Binary fraud label\n\n0 = Legitimate  |  1 = Fraud")

    st.subheader("📊 Independent Variables")
    selected_features = st.multiselect(
        "Select features for model training:",
        options=FEATURES,
        default=FEATURES,
        help="Choose predictor columns"
    )

    st.subheader("🤖 Model")
    model_name = st.selectbox(
        "Algorithm:",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    )

    st.subheader("🔧 Hyperparameters")
    test_size     = st.slider("Test split ratio", 0.10, 0.40, 0.20, 0.05)
    random_state  = st.number_input("Random state", 0, 999, 42)

    train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🔍 Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-end ML pipeline · Explore · Train · Predict</div>', unsafe_allow_html=True)

df = load_data(uploaded)

if len(selected_features) == 0:
    st.warning("⚠️ Please select at least one feature from the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview",
    "📊 EDA",
    "🤖 Model Training",
    "📈 Evaluation",
    "🔮 Predict"
])

# ─── TAB 1 · Data Overview ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Features",        f"{len(FEATURES)}")
    c3.metric("Fraud Cases",     f"{df[TARGET].sum():,}",    f"{df[TARGET].mean()*100:.1f}%")
    c4.metric("Legitimate Cases",f"{(df[TARGET]==0).sum():,}", f"{(1-df[TARGET].mean())*100:.1f}%")

    st.divider()

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("#### Sample Records")
        st.dataframe(df.sample(10, random_state=1).reset_index(drop=True), use_container_width=True)
    with col_r:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df[selected_features].describe().T.round(3), use_container_width=True)

    st.divider()
    st.markdown("#### Feature Descriptions")
    desc_df = pd.DataFrame(
        [(f, FEATURE_DESCRIPTIONS[f], str(df[f].dtype)) for f in FEATURES],
        columns=["Feature", "Description", "Dtype"]
    )
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

# ─── TAB 2 · EDA ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis")

    # Class balance
    col1, col2 = st.columns(2)
    with col1:
        counts = df[TARGET].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            title="Class Distribution",
            color_discrete_sequence=["#2d6ae0", "#e03d2d"]
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        hour_fraud = df.groupby(["Hour", TARGET]).size().reset_index(name="count")
        hour_fraud["label"] = hour_fraud[TARGET].map({0: "Legitimate", 1: "Fraud"})
        fig_hour = px.bar(
            hour_fraud, x="Hour", y="count", color="label",
            title="Transactions by Hour",
            color_discrete_map={"Legitimate": "#2d6ae0", "Fraud": "#e03d2d"},
            barmode="stack"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # Distribution of numeric features
    st.markdown("#### Feature Distributions by Fraud Label")
    num_features = [f for f in selected_features if df[f].nunique() > 10]
    feat_choice  = st.selectbox("Select feature:", num_features[:10] if num_features else selected_features)

    fig_dist = px.histogram(
        df, x=feat_choice, color=df[TARGET].map({0: "Legitimate", 1: "Fraud"}),
        nbins=40, barmode="overlay", opacity=0.7,
        title=f"Distribution of {feat_choice}",
        color_discrete_map={"Legitimate": "#2d6ae0", "Fraud": "#e03d2d"},
        labels={"color": "Label"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap (selected features + target)")
    corr_cols = selected_features[:12] + [TARGET]  # cap for readability
    corr      = df[corr_cols].corr()
    fig_heat, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig_heat, use_container_width=True)
    plt.close()

    # Fraud by day of week
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    dow_fraud = df[df[TARGET]==1]["DayOfWeek"].value_counts().rename(index=dow_map).sort_index()
    fig_dow = px.bar(
        x=dow_fraud.index, y=dow_fraud.values,
        title="Fraud Cases by Day of Week",
        labels={"x": "Day", "y": "Fraud Count"},
        color_discrete_sequence=["#e03d2d"]
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# ─── TAB 3 · Model Training ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Training")

    if train_btn or "model_trained" in st.session_state:
        with st.spinner("Training model…"):
            clf, scaler, metrics, feature_imp = train_model(
                df, selected_features, model_name, test_size, random_state
            )
            st.session_state["model_trained"] = True
            st.session_state["clf"]           = clf
            st.session_state["scaler"]        = scaler
            st.session_state["metrics"]       = metrics
            st.session_state["feature_imp"]   = feature_imp
            st.session_state["sel_features"]  = selected_features

        st.success(f"✅ **{model_name}** trained successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        c2.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")
        c3.metric("F1 Score",  f"{metrics['f1']:.4f}")

        st.divider()
        st.markdown("#### Feature Importances")
        fig_fi = px.bar(
            x=feature_imp.values, y=feature_imp.index, orientation="h",
            title="Feature Importance",
            labels={"x": "Importance", "y": "Feature"},
            color=feature_imp.values,
            color_continuous_scale="Blues"
        )
        fig_fi.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("👈 Configure your model in the sidebar and click **Train Model**.")

# ─── TAB 4 · Evaluation ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Evaluation")

    if "metrics" not in st.session_state:
        st.info("Train a model first (Tab 3).")
    else:
        metrics = st.session_state["metrics"]

        # Confusion matrix
        col1, col2 = st.columns(2)
        with col1:
            cm = metrics["cm"]
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig_cm, use_container_width=True)
            plt.close()

        with col2:
            # ROC Curve
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"], mode="lines",
                name=f"ROC (AUC = {metrics['roc_auc']:.4f})",
                line=dict(color="#2d6ae0", width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                name="Random", line=dict(color="grey", dash="dash")
            ))
            fig_roc.update_layout(
                title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                height=370
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Precision-Recall curve
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=metrics["recall"], y=metrics["precision"], mode="lines",
            name="PR Curve", line=dict(color="#e03d2d", width=2)
        ))
        fig_pr.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall", yaxis_title="Precision", height=350
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        # Classification report
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(metrics["report"]).T.round(4)
        st.dataframe(report_df, use_container_width=True)

# ─── TAB 5 · Predict ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🔮 Single Transaction Prediction")

    if "clf" not in st.session_state:
        st.info("Train a model first (Tab 3).")
    else:
        sel = st.session_state["sel_features"]
        st.markdown("Enter transaction details:")

        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(sel):
            with cols[i % 3]:
                mn  = float(df[feat].min())
                mx  = float(df[feat].max())
                med = float(df[feat].median())
                if df[feat].nunique() <= 10:
                    input_data[feat] = st.selectbox(
                        feat, sorted(df[feat].unique()),
                        index=int(df[feat].mode()[0]),
                        help=FEATURE_DESCRIPTIONS.get(feat, "")
                    )
                else:
                    input_data[feat] = st.number_input(
                        feat, min_value=mn, max_value=mx, value=med,
                        help=FEATURE_DESCRIPTIONS.get(feat, "")
                    )

        if st.button("🔍 Predict", type="primary"):
            X_input = pd.DataFrame([input_data])
            X_scaled = st.session_state["scaler"].transform(X_input)
            pred     = st.session_state["clf"].predict(X_scaled)[0]
            prob     = st.session_state["clf"].predict_proba(X_scaled)[0][1]

            st.divider()
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED** — Probability: {prob*100:.2f}%")
            else:
                st.success(f"✅ **LEGITIMATE** — Fraud Probability: {prob*100:.2f}%")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Fraud Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e03d2d" if pred==1 else "#2d6ae0"},
                    "steps": [
                        {"range": [0,  40], "color": "#d4edda"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70,100], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
    "Fraud Detection Dashboard · Built with Streamlit · Dataset: fraud_detection_cleaned.csv"
    "</p>",
    unsafe_allow_html=True
)
