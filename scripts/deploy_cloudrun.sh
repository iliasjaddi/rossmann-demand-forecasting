#!/usr/bin/env bash
# Deploy the forecast API to Google Cloud Run.
#
#   ./scripts/deploy_cloudrun.sh <PROJECT_ID> [REGION]
#
# ARCHITECTURE NOTE
# Apple Silicon is arm64; Cloud Run runs amd64. An image built natively on this
# Mac and pushed as-is starts and dies instantly with "exec format error".
# Two fixes exist:
#   1. docker buildx build --platform linux/amd64   (QEMU emulation, slow)
#   2. build on Cloud Build, which is already amd64  (what this uses)
# Option 2 is faster, needs no emulation, and is the normal GCP workflow.
# The local image is still worth building for testing; it just isn't the one
# that gets deployed.
set -euo pipefail

PROJECT="${1:?usage: deploy_cloudrun.sh <rossman-demand> [REGION]}"
REGION="${2:-europe-west1}"
SERVICE="rossmann-forecast"
REPO="containers"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}"

echo "project ${PROJECT} | region ${REGION} | service ${SERVICE}"

gcloud config set project "$PROJECT" --quiet

echo "--- enabling required APIs (idempotent)"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --quiet

echo "--- ensuring Artifact Registry repository exists"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" \
    --description="container images" --quiet

echo "--- building amd64 image on Cloud Build"
# Projects created after roughly mid-2024 do not get the legacy Cloud Build
# service account (PROJECT_NUMBER@cloudbuild.gserviceaccount.com) provisioned,
# even though an IAM binding for it still appears in the policy. A plain
# `builds submit` then fails with PERMISSION_DENIED because there is no
# identity to run as. Builds must name a service account explicitly; the
# compute default one exists in every project and carries roles/editor, which
# covers pushing to Artifact Registry and writing logs.
#
# --default-buckets-behavior puts the staging and log buckets in this project,
# which a user-specified service account requires.
#
# What gets uploaded is governed by .gcloudignore. gcloud does NOT read
# .dockerignore.
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
BUILD_SA="projects/${PROJECT}/serviceAccounts/${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "    build identity: ${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud builds submit \
  --tag "$IMAGE" \
  --service-account="$BUILD_SA" \
  --default-buckets-behavior=regional-user-owned-bucket \
  --quiet

echo "--- deploying to Cloud Run"
# --memory 1Gi: the booster plus the 8.2 MB feature table plus pandas and
#   pyarrow sit comfortably under this; 512Mi is tight once pyarrow loads.
# --min-instances 0: scale to zero. An idle demo then costs nothing, at the
#   price of a cold start on the first request after a quiet period.
# --max-instances 3: a public URL on a portfolio has no traffic shape you
#   control. This caps the blast radius of a scraper or a hug of death.
# --allow-unauthenticated: it is a public demo. Do not copy this flag onto
#   anything that returns real data.
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 40 \
  --timeout 60 \
  --min-instances 0 \
  --max-instances 3 \
  --quiet

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" \
        --format 'value(status.url)')
echo
echo "deployed: $URL"
echo
echo "--- smoke testing the deployed service"
./scripts/smoke_test.sh "$URL"
echo
echo "docs: ${URL}/docs"
