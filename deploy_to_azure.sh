#!/bin/bash
# ---------- KONFIG ----------
RG="wunschoele-rg-de"
LOCATION="germanywestcentral"   # <— hier DE-Region eintragen
ACR_NAME="wunschoeleacr75a3"    # bleibt
PLAN_NAME="wunschoele-plan-de2"
APP_NAME="wunschoele-app-de2"
SKU="S1"           
IMAGE_NAME="wunschoele-app:latest"     
# -----------------------------

echo "🔧 RG"
az group create -n $RG -l $LOCATION

echo "🔧 ACR"
az acr create -g $RG -n $ACR_NAME --sku Basic --admin-enabled true
ACR_LOGIN=$(az acr show -g $RG -n $ACR_NAME --query loginServer -o tsv)

echo "🐳 Build & Push"
docker build -t $ACR_LOGIN/$IMAGE_NAME .
az acr login -n $ACR_NAME
docker push $ACR_LOGIN/$IMAGE_NAME

echo "🧱 Plan ($SKU)"
az appservice plan create \
  -g $RG -n $PLAN_NAME \
  --is-linux --sku $SKU \
  --location $LOCATION

echo "🌐 Web-App"
az webapp create \
  -g $RG -p $PLAN_NAME -n $APP_NAME \
  --container-image-name $ACR_LOGIN/$IMAGE_NAME

echo "🔐 Container-Creds"
az webapp config container set \
  -g $RG -n $APP_NAME \
  --container-image-name $ACR_LOGIN/$IMAGE_NAME \
  --container-registry-url https://$ACR_LOGIN \
  --container-registry-user $(az acr credential show -n $ACR_NAME --query username -o tsv) \
  --container-registry-password $(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

echo "🔧 Env"
az webapp config appsettings set -g $RG -n $APP_NAME --settings \
  STREAMLIT_SERVER_HEADLESS=true \
  STREAMLIT_SERVER_PORT=8501 \
  OPENAI_API_KEY=sk-... \
  SUPABASE_URL=https://... \
  SUPABASE_ANNON_KEY=...

echo "🌍 URL:"
az webapp show -g $RG -n $APP_NAME --query defaultHostName -o tsv
