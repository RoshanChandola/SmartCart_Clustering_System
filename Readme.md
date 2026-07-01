# 🛒 Smart Cart Customer Clustering System & Web Dashboard

An end-to-end Machine Learning-powered customer segmentation platform. The system applies unsupervised learning to group customers based on purchasing behavior, demographics, and shopping habits. It features a robust Python **FastAPI backend** pipeline and a premium **glassmorphic dark-mode web dashboard** with interactive 3D visualizations, real-time profile prediction, and bulk CSV customer segmentation.

---

## 📸 Dashboard Preview

- **Interactive 3D PCA Scatter Plot:** Rotate, zoom, and hover over data points in 3D principal component space to examine cluster formations.
- **Segment Explorer:** Detailed trait breakdowns (average income, spending, family size, recency) and custom marketing recommendations for each customer persona.
- **Demographic & Purchase Profiler:** Run real-time predictions to classify new customers and view side-by-side metric comparison charts.
- **Bulk CSV Segmenter:** Upload database CSV files to segment hundreds of customers instantly and download the results as a segmented spreadsheet.

---

## 📂 Folder Structure

```text
SmartCart_Clustering_System/
│
├── backend/
│   ├── main.py             # FastAPI Server (API routes & static file serving)
│   └── model.py            # ML pipeline (Data cleaning, Scaling, PCA, & KNN predictions)
│
├── frontend/
│   ├── index.html          # Web Dashboard Layout
│   ├── css/
│   │   └── styles.css      # Premium Glassmorphic dark styling
│   └── js/
│       └── app.js          # Chart.js, Plotly.js, and client-side logic
│
├── data/
│   ├── smartcart_customers.csv                        # Primary Dataset
│   └── SmartCart Clustering System_project_description.pdf
│
├── models/                 # Saved fit parameters (Generated on training)
│   ├── scaler.joblib       # Fitted StandardScaler
│   ├── pca.joblib          # Fitted PCA reducer
│   ├── knn.joblib          # KNeighborsClassifier (Agglomerative mapping)
│   ├── ohe.joblib          # Fitted OneHotEncoder
│   └── metadata.joblib     # Preprocessing reference metrics
│
├── notebooks/
│   └── smartcart.ipynb     # Original Jupyter development notebook
│
├── visualizations/         # Static evaluation plots
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   ├── customer_clusters.png
│   └── dendrogram.png
│
├── requirements.txt        # Python package list
├── run_app.py              # Root launcher script
└── README.md               # Project documentation
```

---

## 📊 Customer Personas (Clusters)

Through unsupervised Agglomerative Hierarchical Clustering, the dataset is segmented into 4 actionable personas:

| Segment | Icon | Persona Name | Key Behaviors | Actionable Business Strategy |
| :--- | :---: | :--- | :--- | :--- |
| **Cluster 0** | 👨‍👩‍👧‍👦 | Partnered Budget Families | Moderate income, children, low-moderate spend. High web visits but low transactions. | Family multi-buy deals & promo codes to drive web-to-store conversions. |
| **Cluster 1** | 💎 | Partnered VIP Spenders | High income couples, very high spending. Not price sensitive; buy in-store and catalogs. | Direct promotion of luxury products, exclusive customer perks, VIP loyalty. |
| **Cluster 2** | 🛒 | Single Budget Parents | Lowest average income, high children ratio, lowest spend. High web visit frequency. | Targeted clearance campaigns, low-cost convenience items, flexible payment terms. |
| **Cluster 3** | 🎯 | Responsive Single VIPs | High income singles, no children, very high spending. **Highest Campaign Response (32%)**. | Prime targets for promotional campaigns, new product launches, luxury items. |

---

## 🛠️ Tech Stack

- **Machine Learning Pipeline:** Python, Pandas, NumPy, Scikit-learn, Joblib, Kneed
- **Backend API Server:** FastAPI, Uvicorn, Starlette
- **Web UI & Styling:** HTML5, Vanilla CSS (Glassmorphism design, custom grids, CSS transitions)
- **Data Visualizations:** Chart.js (Doughnut charts, Scatter plots, Radar comparisons), Plotly.js (Interactive 3D PCA plot)

---

## 🚀 Installation & How to Run

### 1. Prerequisite Setup
Make sure you are in the project root directory and activate your Python virtual environment.

* **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### 2. Verify Requirements
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 3. Launch the Server
Start the unified FastAPI web app:
```bash
python run_app.py
```

### 4. Open the Web Application
Open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## ⚙️ Model Pipeline Operations

The model training pipeline handles data cleansing and transformation steps sequentially:
1. **Outlier Removal:** Filters out customers with `Age >= 90` and `Income >= $600,000`.
2. **Imputation:** Fills missing `Income` fields with the dataset median value.
3. **Feature Engineering:** Computes customer Age relative to 2026, sign-up tenure in days, total children, and sums up the 6 spending categories into `Total_Spending`.
4. **Encoding & Scaling:** Applies One-Hot Encoding to categorical variables (`Education` and `Living_With`) and scales numerical inputs with a `StandardScaler`.
5. **PCA Reduction:** Reduces data dimensionality to 3 principal components (explaining major variance).
6. **Agglomerative-KNN Classifier:** Clustered points using Ward hierarchical clustering. A KNN model (`k=1`) is fit on the resulting labels to classify single/batch inputs instantly.