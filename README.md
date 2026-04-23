# 🔍 Fraud Detection Dashboard

An interactive **Streamlit** web application for exploring, training, and evaluating machine learning models on a fraud detection dataset.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Dataset** | `fraud_detection_cleaned.csv` (50,000 rows, 21 columns) |
| **Target Variable** | `Fraud_Label` (0 = Legitimate, 1 = Fraud) |
| **Independent Variables** | 20 transaction-level features (see below) |
| **Task** | Binary Classification — Fraud vs Legitimate |

---

## 🗂️ Features Used

| Feature | Description |
|---------|-------------|
| `Transaction_Amount` | Amount of the transaction (₹) |
| `Transaction_Type` | Encoded transaction type |
| `Account_Balance` | Account balance at transaction time |
| `Device_Type` | Device used (encoded) |
| `Location` | Transaction location (encoded) |
| `Merchant_Category` | Merchant category (encoded) |
| `IP_Address_Flag` | Flagged IP address (0/1) |
| `Previous_Fraudulent_Activity` | Prior fraud history (0/1) |
| `Daily_Transaction_Count` | Number of transactions on that day |
| `Avg_Transaction_Amount_7d` | 7-day average transaction amount |
| `Failed_Transaction_Count_7d` | Failed transactions in last 7 days |
| `Card_Type` | Card type (encoded) |
| `Card_Age` | Age of card in months |
| `Transaction_Distance` | Distance from usual location (km) |
| `Authentication_Method` | Auth method used (encoded) |
| `Risk_Score` | Pre-computed risk score |
| `Is_Weekend` | Transaction on weekend (0/1) |
| `Hour` | Hour of transaction (0–23) |
| `DayOfWeek` | Day of week (0 = Monday) |
| `Month` | Month of transaction (1–12) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud-detection-app.git
cd fraud-detection-app
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `fraud_detection_cleaned.csv` in the project root directory (same level as `app.py`).

### 5. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🖥️ App Features

| Tab | Description |
|-----|-------------|
| 📋 **Data Overview** | Dataset shape, sample rows, descriptive statistics, feature descriptions |
| 📊 **EDA** | Class distribution, hourly trends, feature histograms, correlation heatmap |
| 🤖 **Model Training** | Choose algorithm, features, test split; train with one click |
| 📈 **Evaluation** | Confusion matrix, ROC curve, Precision-Recall curve, classification report |
| 🔮 **Predict** | Enter transaction details and get real-time fraud prediction with a gauge chart |

---

## 🤖 Supported Models

- **Random Forest** *(default)*
- **Gradient Boosting**
- **Logistic Regression**

---

## 📁 Project Structure

```
fraud-detection-app/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── fraud_detection_cleaned.csv  # Dataset (add manually)
├── .streamlit/
│   └── config.toml              # Streamlit theme config
├── .gitignore
└── README.md
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (without the CSV — add it to `.gitignore` if it's large).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Connect your GitHub repo, set `app.py` as the entry point.
4. Add the CSV via the **Secrets** or file-upload widget in the app.

---

## 📦 Requirements

- Python 3.9+
- See `requirements.txt` for all packages

---

## 📄 License

MIT License — free to use and modify.
