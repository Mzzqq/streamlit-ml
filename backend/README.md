# 🏦 Bank Marketing Prediction System

Sistem prediksi langganan term deposit bank menggunakan machine learning dengan metodologi CRISP-DM.

## 📋 Deskripsi Proyek

Proyek ini mengembangkan model machine learning untuk memprediksi kemungkinan nasabah bank berlangganan term deposit berdasarkan data demografis dan riwayat kampanye pemasaran. Aplikasi web interaktif dibuat menggunakan Streamlit untuk memudahkan penggunaan model dalam lingkungan bisnis.

## 🎯 Tujuan Bisnis

- **Meningkatkan efektivitas kampanye pemasaran** bank untuk produk term deposit
- **Mengurangi biaya pemasaran** dengan targeting yang lebih tepat
- **Meningkatkan conversion rate** dari 11.7% menjadi target 20%+
- **Mengoptimalkan alokasi sumber daya** tim pemasaran

## 📊 Dataset

- **Sumber**: Bank Marketing Dataset (UCI Machine Learning Repository)
- **Jumlah record**: 45,211 nasabah
- **Fitur**: 16 input features + 1 target variable
- **Target**: Prediksi langganan term deposit (yes/no)
- **Class distribution**: 88.3% No, 11.7% Yes (imbalanced)

### Fitur Dataset:
- **Demografis**: age, job, marital, education
- **Keuangan**: balance, default, housing, loan
- **Kampanye**: contact, duration, campaign, pdays, previous, poutcome, month

## 🔬 Metodologi CRISP-DM

### 1. Business Understanding ✅
- Identifikasi tujuan bisnis: Meningkatkan efektivitas kampanye pemasaran
- Menentukan kriteria sukses: Peningkatan precision dan recall
- Analisis business impact dan ROI

### 2. Data Understanding ✅
- Exploratory Data Analysis (EDA)
- Analisis distribusi fitur dan target
- Identifikasi korelasi dan pola data
- Quality assessment (missing values, outliers)

### 3. Data Preparation ✅
- Feature engineering (age groups, balance categories, duration categories)
- Encoding categorical variables (One-Hot Encoding)
- Scaling numerical features (StandardScaler)
- Class balancing menggunakan SMOTE
- Train-test split (80-20)

### 4. Modeling ✅
- **Algoritma yang diuji**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Model terbaik**: Gradient Boosting
- **Cross-validation** untuk validasi performa

### 5. Evaluation ✅
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Best Model Performance**:
  - Test Accuracy: 88.3%
  - Precision: 50.1%
  - Recall: 71.6%
  - F1-Score: 0.589
  - AUC-ROC: 0.916

### 6. Deployment ✅
- Aplikasi web interaktif menggunakan Streamlit
- API prediksi real-time
- Dashboard analitik
- Model persistence menggunakan joblib

## 🚀 Cara Menjalankan Proyek

### Prerequisites
```bash
Python 3.10+
Git
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd UAS-mobalog

# Install dependencies
pip install -r requirements.txt
```

### Menjalankan Pipeline Machine Learning

1. **Exploratory Data Analysis**:
```bash
python eda_simple.py
```

2. **Data Preparation**:
```bash
python data_preparation.py
```

3. **Model Training**:
```bash
python modeling_fast.py
```

4. **Deploy Streamlit App**:
```bash
streamlit run streamlit_app.py
```

## 📁 Struktur Proyek

```
UAS-mobalog/
├── train.csv                 # Dataset utama
├── eda_simple.py             # Exploratory Data Analysis
├── data_preparation.py       # Data preprocessing dan feature engineering
├── modeling_fast.py          # Model training dan evaluation
├── streamlit_app.py          # Aplikasi web Streamlit
├── requirements.txt          # Dependencies
├── README.md                # Dokumentasi proyek
└── model_files/             # Model dan preprocessor yang tersimpan
    ├── best_model.pkl
    ├── preprocessor.pkl
    ├── label_encoder.pkl
    └── model_info.pkl
```

## 💡 Key Insights

### Business Insights:
- **Durasi panggilan** adalah predictor terkuat (longer calls = higher success)
- **Metode kontak cellular** lebih efektif daripada telephone
- **Previous campaign success** meningkatkan probabilitas secara signifikan
- **Age group 30-60** menunjukkan conversion rate tertinggi

### Technical Insights:
- Dataset imbalanced memerlukan teknik balancing (SMOTE)
- Feature engineering meningkatkan performa model significantly
- Gradient Boosting outperform algoritma lain untuk kasus ini
- Cross-validation penting untuk avoid overfitting

## 🎯 Business Impact

### Current vs Model-Driven Approach:
| Metric | Current Approach | Model-Driven | Improvement |
|--------|------------------|--------------|-------------|
| Calls | 100,000 | 30,000 | -70% |
| Cost | $500,000 | $150,000 | -$350,000 |
| Conversion Rate | 11.7% | 19.6% (projected) | +67% |

## 🌐 Fitur Aplikasi Web

### 🔮 Prediksi
- Input form interaktif untuk data nasabah
- Real-time prediction dengan confidence score
- Visualisasi probabilitas
- Rekomendasi bisnis berdasarkan hasil prediksi

### 📊 Dashboard
- Overview metrics dan KPIs
- Visualisasi distribusi data
- Success rate analysis by categories
- Interactive charts dan graphs

### ℹ️ Model Information
- Performance metrics detail
- Feature importance analysis
- Business interpretation
- Model methodology explanation

## 🔧 Teknologi yang Digunakan

- **Python 3.10**: Programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Imbalanced-learn**: Class balancing (SMOTE)
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Joblib**: Model persistence

## 📈 Future Improvements

1. **Model Enhancement**:
   - Hyperparameter tuning dengan GridSearch/RandomSearch
   - Ensemble methods (Voting, Stacking)
   - Deep learning approaches

2. **Feature Engineering**:
   - Time-based features (seasonality, trends)
   - Interaction features
   - External data integration

3. **Deployment**:
   - Cloud deployment (Heroku, AWS, GCP)
   - CI/CD pipeline
   - Model monitoring dan retraining automation

4. **Business Features**:
   - A/B testing framework
   - Customer segmentation
   - Campaign optimization tools

## 👥 Tim Pengembang

- **Data Scientist**: Analisis data dan modeling
- **ML Engineer**: Pipeline dan deployment
- **Business Analyst**: Business requirements dan insights

## 📝 Lisensi

Project ini dibuat untuk keperluan akademik dan dapat digunakan sesuai dengan [MIT License](LICENSE).

## 📞 Kontak

Untuk pertanyaan atau kolaborasi, silakan hubungi:
- Email: [your-email@domain.com]
- LinkedIn: [your-linkedin-profile]
- GitHub: [your-github-profile]

---

⭐ **Jika proyek ini bermanfaat, jangan lupa berikan star di GitHub!** ⭐ 