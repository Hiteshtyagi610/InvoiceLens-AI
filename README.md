
# 🚀 InvoiceGuard AI  
### Financial Anomaly Detection & Freight Intelligence System  

🔗 **Live App:** https://invoiceguard.streamlit.app/  
📂 **GitHub Repo:** https://github.com/Hiteshtyagi610/InvoiceLens-AI  

---

## 📌 Problem Statement  

In supply chain and financial systems, inaccurate freight estimation and anomalous invoices can lead to significant financial losses.

This project aims to:

- Predict freight costs accurately  
- Detect abnormal or potentially fraudulent invoices  
- Provide a visual analytics system for decision-making  

---

# 🗄️ Dataset & Data Engineering  

The dataset consists of multiple relational tables:

- `purchases`  
- `purchase_prices`  
- `vendor_invoice`  
- `begin_inventory`  
- `end_inventory`  

---

## 🔹 Freight Prediction Data Pipeline  

### Step 1: Table Selection  
- Primary table used: `vendor_invoice`

### Step 2: Exploratory Data Analysis (EDA)  
Performed analysis on:
- Invoice quantity  
- Total price (Dollars)  
- Freight values  

Key observations:
- Freight is highly correlated with invoice price  
- Outliers present in high-value invoices  

---

### Step 3: Feature Selection  

```text
Input Feature → Dollars  
Target → Freight
````

---

### Step 4: Train-Test Split

* Train: 80%
* Test: 20%

---

### Step 5: Model Training

Trained and compared multiple regression models:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

### 📊 Model Performance

#### 🔹 Linear Regression

* MAE: 24.11
* RMSE: 124.72
* R² Score: 96.99%

#### 🔹 Decision Tree

* MAE: 32.97
* RMSE: 150.31
* R² Score: 95.63%

#### 🔹 Random Forest

* MAE: 26.13
* RMSE: 134.79
* R² Score: 96.48%

---

### ✅ Best Model Selected

👉 **Linear Regression**

Reason:

* Highest R² score
* Lowest error
* Simpler and more interpretable

---

# ⚠️ Invoice Risk Detection Pipeline

### Step 1: Data Integration

Combined multiple tables:

* `vendor_invoice`
* `purchases`
* Aggregated purchase-level data

---

### Step 2: Feature Engineering

Created features such as:

* Invoice quantity
* Invoice price
* Freight
* Total purchase quantity
* Aggregated freight values

---

### Step 3: Label Creation

Defined rule-based labeling:

* Large deviation in price vs freight
* Unusual processing delays

```text
flag_invoice = 1 (Risky) / 0 (Normal)
```

---

### Step 4: Feature Scaling

Applied:

* StandardScaler

To normalize feature distribution before training

---

### Step 5: Model Training

Trained classification models:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

### 📊 Model Performance

#### 🔹 Random Forest Classifier

* Accuracy: **89%**

---

### ✅ Best Model Selected

👉 **Random Forest Classifier**

Reason:

* Highest accuracy
* Better handling of non-linear patterns
* Robust to feature variations

---

# 🧠 Model Deployment

Both models were:

* Serialized using **Joblib**
* Integrated into inference pipelines

---

# 🌐 Application (Streamlit)

Built an interactive web application using Streamlit:

### Features:

* 📊 Dashboard with analytics
* 🚚 Freight cost prediction
* ⚠️ Invoice risk detection
* 📂 Bulk data upload (CSV / DB)
* 📈 Visualizations (distribution, scatter, anomaly charts)

---

# 🏗️ System Architecture

```
Database / CSV Input
        ↓
Data Cleaning & Validation
        ↓
Feature Engineering
        ↓
ML Models (Regression + Classification)
        ↓
Prediction Outputs
        ↓
Visualization Dashboard
```

---

# 🧪 Technologies Used

* Python
* Streamlit
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* SQLite
* Joblib

---

# 📈 Key Highlights

* Built end-to-end ML pipeline
* Compared multiple models and selected best
* Combined regression + classification system
* Designed interactive analytics dashboard
* Implemented anomaly detection logic

---

# 🚀 Future Improvements

* Model explainability (SHAP values)
* Real-time data ingestion
* Advanced anomaly detection (Isolation Forest)
* API deployment (FastAPI)
* Interactive visualizations (Plotly)

---

# 👨‍💻 Author

**Hitesh Tyagi**
hiteshtyagi610@gmail.com
+91-7428925864

---

# ⭐ Conclusion

This project demonstrates how machine learning models can be integrated into a real-world system to provide actionable insights in financial and supply chain domains.

```




