import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

class SmartCartClusteringPipeline:
    def __init__(self, models_dir=None):
        if models_dir is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(backend_dir)
            self.models_dir = os.path.join(project_root, "models")
        else:
            self.models_dir = models_dir
        self.scaler = None
        self.pca = None
        self.knn = None
        self.ohe = None
        self.reference_date = None
        self.median_income = None
        self.feature_names = None
        
    def preprocess_df(self, df, is_training=False):
        """
        Performs data cleaning and feature engineering.
        """
        df_copy = df.copy()
        
        # 1. Handle missing Income
        if is_training:
            self.median_income = df_copy["Income"].median()
        df_copy["Income"] = df_copy["Income"].fillna(self.median_income if self.median_income is not None else 50000)
        
        # 2. Age feature
        # Using 2026 to match the Jupyter notebook baseline
        df_copy["Age"] = 2026 - df_copy["Year_Birth"]
        
        # 3. Customer Tenure
        df_copy["Dt_Customer"] = pd.to_datetime(df_copy["Dt_Customer"], dayfirst=True, errors='coerce')
        # If Dt_Customer fails to parse, fill it with reference_date
        if is_training:
            self.reference_date = df_copy["Dt_Customer"].max()
            if pd.isnull(self.reference_date):
                self.reference_date = pd.Timestamp("2014-06-29")
        
        # Fill missing dates with the reference date
        df_copy["Dt_Customer"] = df_copy["Dt_Customer"].fillna(self.reference_date)
        df_copy["Customer_Tenure_Days"] = (self.reference_date - df_copy["Dt_Customer"]).dt.days
        
        # 4. Total Spending
        spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
        # Ensure all spend columns exist (fill NaNs with 0 if they don't, or handle missing)
        for col in spend_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0
        df_copy["Total_Spending"] = df_copy[spend_cols].sum(axis=1)
        
        # 5. Total Children
        df_copy["Total_Children"] = df_copy["Kidhome"] + df_copy["Teenhome"]
        
        # 6. Education mapping
        df_copy["Education"] = df_copy["Education"].replace({
            "Basic": "UnderGraduate", 
            "2n Cycle": "UnderGraduate", 
            "Master": "PostGraduate", 
            "PhD": "PostGraduate"
        })
        
        # 7. Living status mapping
        df_copy["Living_With"] = df_copy["Marital_Status"].replace({
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone",
            "Widow": "Alone"
        })
        
        # 8. Outliers removal (only during training to avoid dropping user requests)
        if is_training:
            df_copy = df_copy[(df_copy["Age"] < 90) & (df_copy["Income"] < 600000)]
            
        return df_copy

    def fit(self, data_path):
        """
        Fits the scaler, PCA, Agglomerative clustering, and KNN classifier on the training data.
        """
        df = pd.read_csv(data_path)
        
        # Clean and pre-process
        df_cleaned = self.preprocess_df(df, is_training=True)
        
        # Categorical columns encoding
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_cols = ["Education", "Living_With"]
        enc_cols = self.ohe.fit_transform(df_cleaned[cat_cols])
        enc_df = pd.DataFrame(enc_cols, columns=self.ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)
        
        # Prepare inputs (dropping raw IDs, dates, and spending sub-columns)
        cols_to_drop = ["ID", "Year_Birth", "Dt_Customer", "Marital_Status", "Kidhome", "Teenhome",
                        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
        df_model_input = df_cleaned.drop(columns=cols_to_drop + cat_cols, errors='ignore')
        
        X = pd.concat([df_model_input, enc_df], axis=1)
        self.feature_names = list(X.columns)
        
        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        self.pca = PCA(n_components=3, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Agglomerative Clustering
        agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
        cluster_labels = agg.fit_predict(X_pca)
        
        # Train KNN classifier to map new points to the Agglomerative clusters
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(X_pca, cluster_labels)
        
        # Save pipeline files
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.joblib"))
        joblib.dump(self.pca, os.path.join(self.models_dir, "pca.joblib"))
        joblib.dump(self.knn, os.path.join(self.models_dir, "knn.joblib"))
        joblib.dump(self.ohe, os.path.join(self.models_dir, "ohe.joblib"))
        joblib.dump({
            "reference_date": self.reference_date,
            "median_income": self.median_income,
            "feature_names": self.feature_names
        }, os.path.join(self.models_dir, "metadata.joblib"))
        
        print("Model pipeline trained and saved successfully.")

    def load(self):
        """
        Loads the pipeline components from saved joblib files.
        """
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        pca_path = os.path.join(self.models_dir, "pca.joblib")
        knn_path = os.path.join(self.models_dir, "knn.joblib")
        ohe_path = os.path.join(self.models_dir, "ohe.joblib")
        metadata_path = os.path.join(self.models_dir, "metadata.joblib")
        
        if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and 
                os.path.exists(knn_path) and os.path.exists(ohe_path) and os.path.exists(metadata_path)):
            raise FileNotFoundError("Pipeline files not found. Please train the model first.")
            
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.knn = joblib.load(knn_path)
        self.ohe = joblib.load(ohe_path)
        
        meta = joblib.load(metadata_path)
        self.reference_date = meta["reference_date"]
        self.median_income = meta["median_income"]
        self.feature_names = meta["feature_names"]

    def transform_single_customer(self, customer_dict):
        """
        Transforms a single customer dictionary into features matching the model input.
        """
        # Convert dictionary to DataFrame (ensure keys map to the columns of raw CSV)
        df_raw = pd.DataFrame([customer_dict])
        
        # Apply preprocessing (which uses saved reference_date and median_income)
        df_processed = self.preprocess_df(df_raw, is_training=False)
        
        # Categorical OHE
        cat_cols = ["Education", "Living_With"]
        enc_cols = self.ohe.transform(df_processed[cat_cols])
        enc_df = pd.DataFrame(enc_cols, columns=self.ohe.get_feature_names_out(cat_cols), index=df_processed.index)
        
        # Drops
        cols_to_drop = ["ID", "Year_Birth", "Dt_Customer", "Marital_Status", "Kidhome", "Teenhome",
                        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
        df_model_input = df_processed.drop(columns=cols_to_drop + cat_cols, errors='ignore')
        
        X = pd.concat([df_model_input, enc_df], axis=1)
        
        # Align features to ensure exact ordering and fill missing OHE columns if any (shouldn't happen with OHE but safe)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
                
        X = X[self.feature_names]
        return X

    def predict_single(self, customer_dict):
        """
        Predicts the customer cluster for a single customer.
        Returns a tuple of (cluster_id, pca_coordinates).
        """
        if self.scaler is None:
            self.load()
            
        X = self.transform_single_customer(customer_dict)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        cluster_id = int(self.knn.predict(X_pca)[0])
        return cluster_id, X_pca[0].tolist()

    def predict_batch(self, df):
        """
        Predicts clusters for a batch of customers in a DataFrame.
        Returns the original DataFrame with a 'predicted_cluster' column.
        """
        if self.scaler is None:
            self.load()
            
        df_processed = self.preprocess_df(df, is_training=False)
        
        cat_cols = ["Education", "Living_With"]
        enc_cols = self.ohe.transform(df_processed[cat_cols])
        enc_df = pd.DataFrame(enc_cols, columns=self.ohe.get_feature_names_out(cat_cols), index=df_processed.index)
        
        cols_to_drop = ["ID", "Year_Birth", "Dt_Customer", "Marital_Status", "Kidhome", "Teenhome",
                        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
        df_model_input = df_processed.drop(columns=cols_to_drop + cat_cols, errors='ignore')
        
        X = pd.concat([df_model_input, enc_df], axis=1)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        clusters = self.knn.predict(X_pca)
        
        result_df = df.copy()
        result_df["predicted_cluster"] = clusters
        return result_df

if __name__ == "__main__":
    # Test training the pipeline
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    data_path = os.path.join(project_root, "data", "smartcart_customers.csv")
    pipeline = SmartCartClusteringPipeline()
    pipeline.fit(data_path)
    pipeline.load()
    
    # Test single prediction
    test_customer = {
        "ID": 9999,
        "Year_Birth": 1980,
        "Education": "Graduation",
        "Marital_Status": "Married",
        "Income": 60000.0,
        "Kidhome": 1,
        "Teenhome": 0,
        "Dt_Customer": "2013-06-15",
        "Recency": 45,
        "MntWines": 300,
        "MntFruits": 50,
        "MntMeatProducts": 200,
        "MntFishProducts": 20,
        "MntSweetProducts": 10,
        "MntGoldProds": 30,
        "NumDealsPurchases": 1,
        "NumWebPurchases": 4,
        "NumCatalogPurchases": 2,
        "NumStorePurchases": 6,
        "NumWebVisitsMonth": 5,
        "Complain": 0,
        "Response": 0
    }
    cluster, coords = pipeline.predict_single(test_customer)
    print(f"Test single customer predicted cluster: {cluster} with PCA coords: {coords}")
