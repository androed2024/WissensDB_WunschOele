#!/bin/bash

export DOCKER_DEFAULT_PLATFORM=linux/amd64


# ---------- KONFIG ----------
RG="wunschoele-rg-de"
ACR_NAME="wunschoeleacr75a3"
APP_NAME="wunschoele-app-de2"
IMAGE_VERSION="v2"
IMAGE_NAME="wunschoele-app:$IMAGE_VERSION"

# -----------------------------

echo "🔐 Hole ACR-Login"
ACR_LOGIN=$(az acr show -g $RG -n $ACR_NAME --query loginServer -o tsv)

echo "🐳 Docker Image bauen"
docker build -t $ACR_LOGIN/$IMAGE_NAME .

echo "🔐 ACR Login & Push"
az acr login -n $ACR_NAME
docker push $ACR_LOGIN/$IMAGE_NAME

echo "🔄 Web App Container-Image aktualisieren"
az webapp config container set \
  -g $RG -n $APP_NAME \
  --container-image-name $ACR_LOGIN/$IMAGE_NAME \
  --container-registry-url https://$ACR_LOGIN \
  --container-registry-user $(az acr credential show -n $ACR_NAME --query username -o tsv) \
  --container-registry-password $(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

echo "✅ Redeploy abgeschlossen!"
echo "🌍 App läuft unter:"
az webapp show -g $RG -n $APP_NAME --query defaultHostName -o tsv
