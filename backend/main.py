import os
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Any

from backend.model import SmartCartClusteringPipeline

app = FastAPI(title="Smart Cart Customer Clustering System API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and Paths (relative to project base)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "smartcart_customers.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

pipeline = SmartCartClusteringPipeline(models_dir=MODELS_DIR)
cached_data = None
cached_stats = None

# Segment Business Insights and Recommendations
SEGMENT_DETAILS = {
    0: {
        "name": "Partnered Budget Shoppers",
        "description": "Middle-income couples or families with children. They have low-to-moderate spending patterns and high web search activity relative to actual purchases. They are careful spenders who look for deals and discounts.",
        "icon": "👨‍👩‍👧‍👦",
        "color": "#9b5de5", # Vibrant Purple
        "income_level": "Moderate (~$39,680)",
        "spending_level": "Low/Moderate (~$222)",
        "children": "Yes (~1.24 children)",
        "living_status": "Partnered",
        "strategies": [
            "Provide family-oriented discounts and multi-buy package deals.",
            "Run price-drop alerts and promo code campaigns to drive web-to-store conversions.",
            "Highlight high-value, cost-effective alternatives in product searches."
        ]
    },
    1: {
        "name": "Partnered VIP Spenders",
        "description": "High-income couples or families without children. They spend very heavily across all categories, especially on premium products, and prefer shopping directly through store and catalog channels. They are not very price-sensitive.",
        "icon": "💎",
        "color": "#00f5d4", # Neon Cyan/Teal
        "income_level": "High (~$72,808)",
        "spending_level": "Very High (~$1,236)",
        "children": "Few/No (~0.51 children)",
        "living_status": "Partnered",
        "strategies": [
            "Promote premium product catalogs (e.g., fine wines, high-end meats).",
            "Offer a VIP loyalty program, exclusive customer service lines, and store reservation perks.",
            "Create high-value product bundle recommendations for couples."
        ]
    },
    2: {
        "name": "Single Budget Shoppers",
        "description": "Lower-income single parents. They have the lowest average spending and highest children ratio. They visit the online store very frequently but rarely complete high-ticket transactions. They are highly price-conscious.",
        "icon": "🛒",
        "color": "#ff007f", # Bright Pink/Magenta
        "income_level": "Low (~$36,960)",
        "spending_level": "Lowest (~$166)",
        "children": "High (~1.27 children)",
        "living_status": "Alone",
        "strategies": [
            "Deploy highly target-specific budget deals and clearance campaigns.",
            "Promote low-cost food, kids items, and high-frequency basic commodities.",
            "Offer flexible payment terms, loyalty points, or free shipping threshold updates."
        ]
    },
    3: {
        "name": "Single VIP Spenders",
        "description": "High-income singles without children. They are big spenders, shop frequently online and in-store, and are extremely responsive to marketing campaigns (32% acceptance rate—highest across all segments). They are the primary targets for promotions.",
        "icon": "🎯",
        "color": "#00bbf9", # Sky Blue
        "income_level": "High (~$70,722)",
        "spending_level": "Very High (~$1,190)",
        "children": "Lowest (~0.46 children)",
        "living_status": "Alone",
        "strategies": [
            "Target with new product launches and marketing campaigns immediately (high conversion rate).",
            "Promote luxury items, gourmet food, gold/specialty goods, and single-serve convenience products.",
            "Use email marketing and personalized web recommendations for quick conversions."
        ]
    }
}

class CustomerInput(BaseModel):
    Year_Birth: int
    Education: str
    Marital_Status: str
    Income: float
    Kidhome: int
    Teenhome: int
    Dt_Customer: str  # YYYY-MM-DD
    Recency: int
    MntWines: float
    MntFruits: float
    MntMeatProducts: float
    MntFishProducts: float
    MntSweetProducts: float
    MntGoldProds: float
    NumDealsPurchases: int
    NumWebPurchases: int
    NumCatalogPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int
    Complain: int
    Response: int

def load_and_cache_dataset():
    global cached_data, cached_stats
    try:
        pipeline.load()
    except FileNotFoundError:
        # Train model if not exists
        pipeline.fit(DATA_PATH)
        pipeline.load()
        
    df = pd.read_csv(DATA_PATH)
    # Batch predict cluster labels
    cached_data = pipeline.predict_batch(df)
    
    # Calculate stats
    # 1. Total overview metrics
    total_customers = len(cached_data)
    avg_income = float(cached_data["Income"].fillna(cached_data["Income"].median()).mean())
    
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    total_spend = cached_data[spend_cols].sum(axis=1)
    avg_spend = float(total_spend.mean())
    
    campaign_response_rate = float(cached_data["Response"].mean() * 100)
    
    # 2. Cluster breakdowns
    cluster_counts = cached_data["predicted_cluster"].value_counts().to_dict()
    # Ensure all clusters exist in keys
    for i in range(4):
        if i not in cluster_counts:
            cluster_counts[i] = 0
            
    # Means per cluster
    df_cleaned_with_cluster = pipeline.preprocess_df(df, is_training=False)
    df_cleaned_with_cluster["cluster"] = cached_data["predicted_cluster"]
    
    summary_cols = ["Age", "Income", "Total_Spending", "Total_Children", "Recency", 
                    "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases", "NumWebVisitsMonth", "Response"]
    cluster_means = df_cleaned_with_cluster.groupby("cluster")[summary_cols].mean().to_dict(orient="index")
    
    # Fill in categorical modes or breakdowns per cluster
    for c in range(4):
        cluster_df = df_cleaned_with_cluster[df_cleaned_with_cluster["cluster"] == c]
        if len(cluster_df) > 0:
            edu_mode = cluster_df["Education"].mode()[0]
            living_mode = cluster_df["Living_With"].mode()[0]
        else:
            edu_mode = "Graduation"
            living_mode = "Alone"
            
        if c not in cluster_means:
            cluster_means[c] = {col: 0.0 for col in summary_cols}
            
        cluster_means[c]["Education"] = edu_mode
        cluster_means[c]["Living_With"] = living_mode
        cluster_means[c]["size"] = int(cluster_counts[c])
        cluster_means[c]["percentage"] = float(cluster_counts[c] / total_customers * 100)
    
    # 3. PCA points for visualization (Limit to 1500 points to keep load times snappy)
    # Scaled and PCA components
    df_preprocessed = pipeline.preprocess_df(df, is_training=False)
    
    cat_cols = ["Education", "Living_With"]
    enc_cols = pipeline.ohe.transform(df_preprocessed[cat_cols])
    enc_df = pd.DataFrame(enc_cols, columns=pipeline.ohe.get_feature_names_out(cat_cols), index=df_preprocessed.index)
    
    cols_to_drop = ["ID", "Year_Birth", "Dt_Customer", "Marital_Status", "Kidhome", "Teenhome",
                    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_model_input = df_preprocessed.drop(columns=cols_to_drop + cat_cols, errors='ignore')
    
    X = pd.concat([df_model_input, enc_df], axis=1)
    X = X[pipeline.feature_names]
    
    X_scaled = pipeline.scaler.transform(X)
    X_pca = pipeline.pca.transform(X_scaled)
    
    pca_points = []
    # Sample to speed up Plotly rendering if necessary, but 2200 is fast enough
    for i in range(len(df)):
        pca_points.append({
            "id": int(df.iloc[i]["ID"]),
            "x": float(X_pca[i][0]),
            "y": float(X_pca[i][1]),
            "z": float(X_pca[i][2]),
            "cluster": int(cached_data.iloc[i]["predicted_cluster"]),
            "income": float(df_preprocessed.iloc[i]["Income"]),
            "spending": float(df_preprocessed.iloc[i]["Total_Spending"]),
            "age": int(df_preprocessed.iloc[i]["Age"])
        })
        
    cached_stats = {
        "overview": {
            "total_customers": total_customers,
            "avg_income": avg_income,
            "avg_spending": avg_spend,
            "response_rate": campaign_response_rate
        },
        "clusters": cluster_means,
        "pca_points": pca_points
    }

@app.on_event("startup")
def startup_event():
    load_and_cache_dataset()

@app.get("/api/stats")
def get_stats():
    if cached_stats is None:
        load_and_cache_dataset()
    return cached_stats

@app.post("/api/predict")
def predict_customer(customer: CustomerInput):
    try:
        # Convert Pydantic object to dict
        cust_dict = customer.dict()
        cluster_id, pca_coords = pipeline.predict_single(cust_dict)
        
        # Calculate comparison details
        cluster_info = SEGMENT_DETAILS[cluster_id]
        
        # We can construct a simple comparison object
        total_spend = sum([
            customer.MntWines, customer.MntFruits, customer.MntMeatProducts,
            customer.MntFishProducts, customer.MntSweetProducts, customer.MntGoldProds
        ])
        
        age = 2026 - customer.Year_Birth
        
        comparison = {
            "customer": {
                "Income": customer.Income,
                "Total_Spending": total_spend,
                "Age": age,
                "Total_Children": customer.Kidhome + customer.Teenhome,
                "Recency": customer.Recency
            },
            "cluster_averages": {
                "Income": cached_stats["clusters"][cluster_id]["Income"],
                "Total_Spending": cached_stats["clusters"][cluster_id]["Total_Spending"],
                "Age": cached_stats["clusters"][cluster_id]["Age"],
                "Total_Children": cached_stats["clusters"][cluster_id]["Total_Children"],
                "Recency": cached_stats["clusters"][cluster_id]["Recency"]
            }
        }
        
        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster_info["name"],
            "cluster_details": cluster_info,
            "pca_coords": pca_coords,
            "comparison": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Basic validation of required columns
        required_cols = ["Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome", "Dt_Customer", "Recency"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"CSV is missing columns: {', '.join(missing_cols)}")
            
        result_df = pipeline.predict_batch(df)
        
        # Calculate summary of the batch upload
        batch_counts = result_df["predicted_cluster"].value_counts().to_dict()
        for i in range(4):
            if i not in batch_counts:
                batch_counts[i] = 0
                
        # Prepare output rows
        records = []
        for i in range(min(500, len(result_df))): # Limit return records to 500 for display
            row = result_df.iloc[i]
            records.append({
                "ID": int(row["ID"]) if "ID" in result_df.columns else i,
                "Age": int(2026 - row["Year_Birth"]),
                "Income": float(row["Income"]) if not pd.isna(row["Income"]) else None,
                "Education": str(row["Education"]),
                "Living_With": "Partner" if str(row["Marital_Status"]) in ["Married", "Together"] else "Alone",
                "Total_Children": int(row["Kidhome"] + row["Teenhome"]),
                "Recency": int(row["Recency"]),
                "Cluster": int(row["predicted_cluster"]),
                "Cluster_Name": SEGMENT_DETAILS[int(row["predicted_cluster"])]["name"]
            })
            
        # Write full results to a temporary stream for download
        output_stream = io.StringIO()
        result_df.to_csv(output_stream, index=False)
        csv_content = output_stream.getvalue()
        
        return {
            "total_records": len(result_df),
            "counts": {
                "0": int(batch_counts[0]),
                "1": int(batch_counts[1]),
                "2": int(batch_counts[2]),
                "3": int(batch_counts[3])
            },
            "segment_meta": SEGMENT_DETAILS,
            "preview": records,
            "full_csv_raw": csv_content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.post("/api/retrain")
def retrain_model():
    try:
        pipeline.fit(DATA_PATH)
        load_and_cache_dataset()
        return {"status": "success", "message": "Model pipeline retrained and cached successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

# Mount static files (will serve frontend directory)
# Make sure frontend directories exist
frontend_dir = os.path.join(BASE_DIR, "frontend")
os.makedirs(os.path.join(frontend_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(frontend_dir, "js"), exist_ok=True)

app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
