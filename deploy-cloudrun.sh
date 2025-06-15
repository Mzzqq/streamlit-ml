#!/bin/bash

# Cloud Run Deployment Script
# Make sure you have gcloud CLI installed and authenticated

# Configuration
PROJECT_ID="streamlit-ml-463009"
SERVICE_NAME="bank-ml-api"
REGION="asia-southeast1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting Cloud Run deployment..."

# Build and push image to Container Registry
echo "📦 Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo "🌐 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "ENVIRONMENT=production"

echo "✅ Deployment completed!"
echo "🔗 Your service URL:"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)" 