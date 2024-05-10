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

set -o xtrace
set -e

if [ -z "${MODEL_BUCKET}" ]; then
    echo "Error: The environment variable 'MODEL_BUCKET' is not set."
    exit 1
fi

if [ -z "${DATASET_BUCKET}" ]; then
    echo "Error: The environment variable 'DATASET_BUCKET' is not set."
    exit 1
fi

TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
DATASET='horses_or_humans'
MODEL_DIR="gs://$MODEL_BUCKET/$DATASET/$DATASET-model-$TIMESTAMP"
TRAIN_PATTERN="gs://$DATASET_BUCKET/$DATASET/3.0.0/$DATASET-train.tfrecord*"
VAL_PATTERN="gs://$DATASET_BUCKET/$DATASET/3.0.0/$DATASET-test.tfrecord*"
BATCH_SIZE=32
TRAIN_STEPS=100
VAL_STEPS=10
LEARNING_RATE=0.001
MOMENTUM=0.9
UNITS=32
EPOCHS=2
STRATEGY="ParameterServerStrategy"

# Check if the Python script exists
if [[ ! -f trainer/task.py ]]; then
    echo "Error: Python script 'trainer/task.py' not found."
    exit 1
fi

# Call the Python script with variables embedded
python trainer/task.py train \
  --model_dir "$MODEL_DIR" \
  --train_pattern "$TRAIN_PATTERN" \
  --val_pattern "$VAL_PATTERN" \
  --batch_size "$BATCH_SIZE" \
  --train_steps "$TRAIN_STEPS" \
  --val_steps "$VAL_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --momentum "$MOMENTUM" \
  --units "$UNITS" \
  --epochs "$EPOCHS" \
  --strategy "$STRATEGY" \
  in_process \
  --num_workers 4 \
  --num_ps 1
