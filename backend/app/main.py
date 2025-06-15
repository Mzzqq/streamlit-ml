# backend/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting term deposit subscriptions based on customer data and providing analytics.",
    version="1.0.0"
)

# Konfigurasi CORS
origins = [
    "http://localhost:3000",
    #"https://frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definisikan jalur ke file model dan data.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "train.csv") # Path ke dataset

# Variabel global untuk menyimpan model, preprocessor, dan data
model = None
preprocessor = None
label_encoder = None
model_info = None
df_global = None # Untuk menyimpan DataFrame yang dimuat

# Fungsi untuk memuat model, preprocessor, dan data
def load_resources_from_disk():
    """Load trained model, preprocessor, label encoder, model info, and dataset."""
    global model, preprocessor, label_encoder, model_info, df_global
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        try:
            model_info_loaded = joblib.load(os.path.join(MODEL_DIR, "model_info.pkl"))
            model_info = model_info_loaded
        except FileNotFoundError:
            print(f"Warning: model_info.pkl not found at {os.path.join(MODEL_DIR, 'model_info.pkl')}. Using default info.")
            model_info = {} 
        except Exception as e:
            print(f"Error loading model_info.pkl: {e}")
            model_info = {}

        default_model_metrics = {
            'accuracy': 0.883,
            'precision': 0.501,
            'recall': 0.716,
            'F1-Score': 0.589,
            'AUC-ROC': 0.916,
            'CV Folds': 3,
            'CV Score': '0.465 \u00B1 0.023',
        }


        for key, default_value in default_model_metrics.items():
            if key not in model_info:
                model_info[key] = default_value
    
        if 'model_name' not in model_info:
            model_info['model_name'] = 'Gradient Boosting'
        if 'model_type' not in model_info:
            model_info['model_type'] = 'GradientBoostingClassifier'


        # Load the dataset for analytics
        df_global = pd.read_csv(DATA_PATH, delimiter=';')
        
        print("Model, preprocessor, label encoder, model info (augmented), and dataset loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure all .pkl files are in '{MODEL_DIR}' and '{DATA_PATH}' exists.")
        print(f"Missing file: {e.filename}")
        raise HTTPException(status_code=500, detail=f"File not found: {e.filename}")
    except Exception as e:
        print(f"Error loading resources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load required components: {e}")

# Panggil fungsi load saat aplikasi dimulai
@app.on_event("startup")
async def startup_event():
    load_resources_from_disk()

# Pydantic model untuk input prediksi
class BankCustomerInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: float
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# Fungsi feature engineering
def engineer_features(input_data: dict) -> dict:
    """Perform feature engineering on the input data."""
    engineered_data = input_data.copy()

    # Create derived features
    engineered_data['was_contacted_before'] = 1 if engineered_data['pdays'] != -1 else 0
    engineered_data['pdays_clean'] = 0 if engineered_data['pdays'] == -1 else engineered_data['pdays']
    
    # Age groups
    age = engineered_data['age']
    if age <= 30:
        age_group = '18-30'
    elif age <= 40:
        age_group = '31-40'
    elif age <= 50:
        age_group = '41-50'
    elif age <= 60:
        age_group = '51-60'
    else:
        age_group = '60+'
    engineered_data['age_group'] = age_group
    
    # Balance categories
    balance = engineered_data['balance']
    if balance < 0:
        balance_category = 'negative'
    elif balance <= 1000:
        balance_category = 'low'
    elif balance <= 5000:
        balance_category = 'medium'
    else:
        balance_category = 'high'
    engineered_data['balance_category'] = balance_category
    
    # Duration categories
    duration = engineered_data['duration']
    if duration <= 120:
        duration_category = 'very_short'
    elif duration <= 300:
        duration_category = 'short'
    elif duration <= 600:
        duration_category = 'medium'
    else:
        duration_category = 'long'
    engineered_data['duration_category'] = duration_category
    
    # Binary features
    engineered_data['contact_cellular'] = 1 if engineered_data['contact'] == 'cellular' else 0
    engineered_data['prev_success'] = 1 if engineered_data['poutcome'] == 'success' else 0
    
    return engineered_data

# Endpoint prediksi (tetap sama)
@app.post("/predict")
async def predict_deposit(customer_data: BankCustomerInput):
    if model is None or preprocessor is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Komponen model belum dimuat. Silakan coba lagi sebentar.")

    input_dict = customer_data.dict()
    engineered_data = engineer_features(input_dict)
    
    feature_order = [
        'age', 'balance', 'duration', 'campaign', 'pdays_clean', 'previous',
        'job', 'marital', 'education', 'default', 'housing', 'loan', 
        'contact', 'month', 'poutcome', 'age_group', 'balance_category', 
        'duration_category', 'was_contacted_before', 'contact_cellular', 'prev_success'
    ]
    
    try:
        df_input = pd.DataFrame([engineered_data])[feature_order]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Fitur hilang dalam data input: {e}. Pastikan semua fitur yang diperlukan ada dan dinamai dengan benar setelah rekayasa fitur.")
    
    try:
        X_processed = preprocessor.transform(df_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kesalahan selama pra-pemrosesan data. Pastikan format data input sesuai dengan skema pelatihan. Detail: {e}")
    
    prediction_raw = model.predict(X_processed)[0]
    probability_raw = model.predict_proba(X_processed)[0]
    
    prediction_label = label_encoder.inverse_transform([prediction_raw])[0]
    probability_yes = probability_raw[1]

    return {
        "prediction": prediction_label,
        "probability_yes": float(probability_yes)
    }

# Endpoint dasar (tetap sama)
@app.get("/")
async def root():
    return {"message": "Bank Marketing Prediction API is running!"}

# --- NEW: Endpoints for Chart Data ---

# Endpoint: Distribusi Target Nasabah (Pie Chart)
@app.get("/api/dashboard/target-distribution")
async def get_target_distribution():
    if df_global is None:
        raise HTTPException(status_code=503, detail="Dataset belum dimuat.")
    
    distribution = df_global['y'].value_counts().reset_index()
    distribution.columns = ['label', 'value']
    
    distribution['label'] = distribution['label'].map({'yes': 'Berlangganan', 'no': 'Tidak Berlangganan'})
    
    return distribution.to_dict(orient='records')

# Endpoint: Success Rate by Job Category (Bar Chart)
@app.get("/api/dashboard/job-success-rate")
async def get_job_success_rate():
    if df_global is None:
        raise HTTPException(status_code=503, detail="Dataset belum dimuat.")
    
    job_success = df_global.groupby('job')['y'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
    job_success.columns = ['job', 'success_rate']
    job_success = job_success.sort_values('success_rate', ascending=True)
    
    job_mapping = {
        'admin.': 'Administrasi', 'blue-collar': 'Kerah Biru', 'entrepreneur': 'Wiraswasta',
        'housemaid': 'Pembantu Rumah Tangga', 'management': 'Manajemen', 'retired': 'Pensiunan',
        'self-employed': 'Wirausaha', 'services': 'Pelayanan', 'student': 'Pelajar',
        'technician': 'Teknisi', 'unemployed': 'Pengangguran', 'unknown': 'Tidak Diketahui'
    }
    job_success['job'] = job_success['job'].map(job_mapping)

    return job_success.to_dict(orient='records')

# Endpoint: Distribusi Usia (Histogram/Bar Chart)
@app.get("/api/insights/age-distribution")
async def get_age_distribution():
    if df_global is None:
        raise HTTPException(status_code=503, detail="Dataset belum dimuat.")

    df_temp = df_global.copy()
    
    def get_age_group(age):
        if age <= 30: return '18-30'
        elif age <= 40: return '31-40'
        elif age <= 50: return '41-50'
        elif age <= 60: return '51-60'
        else: return '60+'

    df_temp['age_group'] = df_temp['age'].apply(get_age_group)
    
    age_distribution = df_temp.groupby(['age_group', 'y']).size().unstack(fill_value=0).reset_index()
    age_distribution.columns.name = None 
    age_distribution = age_distribution.rename(columns={'no': 'Tidak Berlangganan', 'yes': 'Berlangganan'})

    age_group_order = ['18-30', '31-40', '41-50', '51-60', '60+']
    age_distribution['age_group'] = pd.Categorical(age_distribution['age_group'], categories=age_group_order, ordered=True)
    age_distribution = age_distribution.sort_values('age_group')

    return age_distribution.to_dict(orient='records')

# Endpoint: Sampel Data untuk Scatter Plot (Balance vs Duration)
@app.get("/api/insights/balance-duration-sample")
async def get_balance_duration_sample():
    if df_global is None:
        raise HTTPException(status_code=503, detail="Dataset belum dimuat.")
    
    sample_df = df_global.sample(n=1000, random_state=42)[['balance', 'duration', 'y']]
    
    sample_df['y_label'] = sample_df['y'].map({'yes': 'Berlangganan', 'no': 'Tidak Berlangganan'})

    return sample_df.to_dict(orient='records')

# --- NEW: Endpoint for Model Info ---
@app.get("/api/model-info")
async def get_model_info():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Informasi model belum dimuat.")
    
    # Mengembalikan objek model_info secara langsung
    return model_info
