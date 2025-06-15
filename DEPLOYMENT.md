# 🚀 Panduan Deployment ke Google Cloud Run

## Prasyarat

1. **Google Cloud CLI terinstall**
   ```bash
   # Install gcloud CLI
   # https://cloud.google.com/sdk/docs/install
   ```

2. **Autentikasi Google Cloud**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable APIs yang diperlukan**
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

## 📋 Checklist Sebelum Deploy

- [ ] File model tersedia di `backend/model_files/`:
  - [ ] `best_model.pkl`
  - [ ] `preprocessor.pkl`
  - [ ] `label_encoder.pkl`
  - [ ] `model_info.pkl`
- [ ] File `train.csv` tersedia di `backend/`
- [ ] Update `PROJECT_ID` di `deploy-cloudrun.sh`
- [ ] Dockerfile sudah benar ✅
- [ ] Requirements.txt lengkap ✅

## 🔧 Konfigurasi

### 1. Update Project ID
Edit file `deploy-cloudrun.sh`:
```bash
PROJECT_ID="your-actual-project-id"
```

### 2. (Opsional) Update Region
Default region: `asia-southeast2` (Jakarta)
Bisa diganti ke region lain jika diperlukan.

## 🚀 Deployment

### Opsi 1: Manual Deploy (Recommended untuk testing)
```bash
# 1. Build dan push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/bank-ml-api .

# 2. Deploy ke Cloud Run
gcloud run deploy bank-ml-api \
    --image gcr.io/YOUR_PROJECT_ID/bank-ml-api \
    --platform managed \
    --region asia-southeast2 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10
```

### Opsi 2: Menggunakan Script
```bash
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh
```

### Opsi 3: Automated CI/CD dengan Cloud Build
Push code ke repository yang terhubung dengan Cloud Build:
```bash
git add .
git commit -m "Deploy to Cloud Run"
git push origin master
```

## 🔍 Verification

Setelah deployment berhasil:

1. **Check service status**
   ```bash
   gcloud run services list
   ```

2. **Get service URL**
   ```bash
   gcloud run services describe bank-ml-api --region=asia-southeast2 --format="value(status.url)"
   ```

3. **Test API endpoints**
   ```bash
   # Test root endpoint
   curl https://YOUR_SERVICE_URL/

   # Test model info
   curl https://YOUR_SERVICE_URL/api/model-info

   # Test prediction
   curl -X POST https://YOUR_SERVICE_URL/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 35,
       "job": "admin.",
       "marital": "married",
       "education": "secondary",
       "balance": 1000,
       "default": "no",
       "housing": "yes",
       "loan": "no",
       "contact": "cellular",
       "month": "may",
       "duration": 180,
       "campaign": 2,
       "pdays": -1,
       "previous": 0,
       "poutcome": "unknown"
     }'
   ```

## 🐛 Troubleshooting

### Error: File model tidak ditemukan
```
Error: Required file not found. Ensure all .pkl files are in 'model_files'
```
**Solusi**: Pastikan semua file model ada di `backend/model_files/`

### Error: Port 8080 tidak tersedia
**Solusi**: Dockerfile sudah dikonfigurasi dengan port 8080, pastikan tidak ada konflik

### Error: Memory insufficient
**Solusi**: Tingkatkan memory allocation di script deployment:
```bash
--memory 2Gi  # atau sesuai kebutuhan
```

### CORS Error
**Solusi**: CORS sudah dikonfigurasi untuk mengizinkan semua origin. Jika masih error, cek logs:
```bash
gcloud logs read --service=bank-ml-api --limit=50
```

## 📊 Monitoring

- **Cloud Console**: https://console.cloud.google.com/run
- **Logs**: `gcloud logs read --service=bank-ml-api`
- **Metrics**: Available in Cloud Console

## 💰 Cost Optimization

- Service hanya charged saat ada request
- Auto-scaling dari 0 instance
- Max instances: 10 (bisa disesuaikan)
- Memory: 1Gi (bisa disesuaikan sesuai kebutuhan)

## 🔒 Security

- Service deployed dengan `--allow-unauthenticated`
- Untuk production, consider menggunakan IAM authentication
- HTTPS termination otomatis disediakan oleh Cloud Run 