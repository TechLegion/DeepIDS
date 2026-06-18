# DeepIDS v2.0: Advanced Network Intrusion Detection System

DeepIDS is a machine learning-based Network Intrusion Detection System (NIDS) designed to classify network packet telemetry and identify cybersecurity threats in real-time. Built using the **NSL-KDD dataset**, the system employs a highly optimized **Gradient Boosting Classifier** to execute multi-class threat classification across five distinct categories.

---

## 🚀 Key Features

* **5-Class Cyber-Threat Classification:** Categorizes traffic as either **Normal** or one of four major attack classes:
  * **DoS (Denial of Service):** High-volume exhaustion attacks (e.g., Neptune, Smurf).
  * **Probe:** Systematic reconnaissance and network scanning (e.g., Satan, Portsweep).
  * **R2L (Remote to Local):** Unauthorized remote access attempts (e.g., Guess Password, Warezclient).
  * **U2R (User to Root):** Privilege escalation attacks attempting to gain admin permissions.
* **SMOTE Class Imbalance Resolution:** Implements Synthetic Minority Over-sampling Technique (SMOTE) to prevent model bias against rare attacks (like U2R and R2L).
* **Interactive Threat Dashboard:** A clean, dark-themed responsive dashboard for real-time diagnostic testing.
* **Batch Traffic Analyzer:** Allows network security personnel to upload a CSV file containing connection records for instant, bulk network auditing.
* **Real-time Probability Analysis:** Displays a dynamic confidence rating and threat-level probability breakdown using Chart.js.

---

## 📂 Project Structure

```bash
├── backend.py                   # FastAPI backend server
├── index.html                   # Threat Intelligence Dashboard (Frontend UI)
├── script.js                    # Interactive dashboard logic & API client
├── style.css                    # Dashboard styling (Premium dark mode UI)
├── train_model.py               # Model training and artifact saving pipeline
├── download_dataset.py          # Script to fetch/prepare the NSL-KDD dataset
├── generate_charts.py           # Dataset distribution visualization script
├── generate_imbalance_charts.py # Class imbalance and SMOTE distribution visualizations
├── generate_eval_chart.py       # Deployed model evaluation vs benchmark models chart
├── requirements.txt             # Python dependencies
└── models/                      # Saved trained models, encoders, and scalers
    ├── gb_model.pkl             # Deployed Gradient Boosting model
    ├── scaler.pkl               # Standard scaled feature transformer
    ├── class_names.pkl          # Target categorical class labels
    └── le_*.pkl                 # Fitted label encoders for protocol, service, flag
```

---

## 📊 Model Performance

Evaluated on the true, unsampled test split of the NSL-KDD dataset (25,195 connection records):

* **Accuracy:** **`99.87%`**
* **Weighted F1-Score:** **`99.88%`**

### Class-Specific Detection Performance
* **DoS:** `99.98%` F1-Score
* **Normal Traffic:** `99.88%` F1-Score
* **Probe / Scan:** `99.70%` F1-Score
* **Remote-to-Local (R2L):** `97.52%` F1-Score
* **User-to-Root (U2R):** `80.00%` F1-Score

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Clone and Initialize Virtual Environment
Open your command terminal in the project root directory and run:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Fetch Dataset and Train Model
```bash
# Download the dataset files
python download_dataset.py

# Train the model and save artifacts to /models
python train_model.py
```

### 4. Run the Web Dashboard
```bash
# Start the FastAPI + Uvicorn server
python -m uvicorn backend:app --reload
```

Once running, open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📈 Generating System Visualizations
To update or generate the academic thesis visualization figures, run:
```bash
# Generate Dataset overview distribution charts
python generate_charts.py

# Generate Class imbalance & SMOTE analysis charts
python generate_imbalance_charts.py

# Generate model comparison and evaluation metrics
python generate_eval_chart.py
```
*Generated plots will be saved in the root folder as `.png` images.*
