#!/bin/bash -eu
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build and push Docker image to Artifact Registry

set -o xtrace

PROJECT_ID="$(gcloud config get-value project)"
REGION="$(gcloud config get-value compute/region)"
AR_REPOSITORY='vertex-pss-demo'

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/vertex-pss:latest"

gcloud builds submit \
    --project=$PROJECT_ID \
    --region=$REGION \
    --tag=$IMAGE_URI

echo "Image pushed to Artifact Registry: $IMAGE_URI"
