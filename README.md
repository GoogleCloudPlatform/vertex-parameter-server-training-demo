# Vertex AI Custom Training with Tensorflow ParameterServerStrategy demo

This project demostrates how to asynchronous model training on Cloud Vertex AI
with TensorFlow ParameterServerStrategy.

## Folders

- `scripts`: Scripts to build and push Docker images, and run model training
- `trainer`: The model training application


## Prerequisites

- Python 3
- Cloud SDK (i.e. `gcloud`)

## Usage

### Configure GCP project

Make sure you have a Google Cloud project and have authenticated your credentials.
```
gcloud config set project <my-gcp-project>
gcloud auth login
```

You will need to create at least one bucket to store the data and the models.
However, having separate buckets for these types of data would be preferred.

For example:
```
gs://my-gcp-project-models
gs://my-gcp-project-datasets
```

### Prepare the dataset
The dataset you will use is `horses_or_humans` from the TensorFlow Datasets (TFDS)
catalog. If you prefer, you can use another dataset from the same catalog that used
for image classification.

Run the following script to upload the dataset to your GCS bucket:
```
python scripts/prepare_dataset.py \
    horses_or_numans \
    gs://my-gcp-project-datasets
```

Check that the dataset was uploaded:
```
gsutil ls gs://my-gcp-project-datasets/horses_or_humans
```

The above path may contain a directory specifying the dataset version, e.g.
`3.0.0`. Under that directory should be the data files in TFRecord format, e.g.
`horses_or_humans-train.tfrecord*`.


### Training the model locally

Before running model training on Vertex AI, it may be helpful to check model
training on your local machine.

Define the `MODELS_BUCKET` and `DATASETS_BUCKET` environmental variables to
specify the GCS buckets for your models and datasets respectively.

For example:
```
export MODELS_BUCKET='gs://my-gcp-project-models'
export DATASETS_BUCKET='gs://my-gcp-project-datasets'
```

The following script will run an in-process cluster with TensorFlow
ParameterServerStrategy. The different task servers (i.e. chief, worker, ps)
will be run as separate processes on the local machine.

```bash
bash scripts/local_train.sh
```

The training logic can be found in `trainer/task.py`.


### Train the model in Vertex AI

#### Build the trainer image

You would need to prepare the model training application as a Docker
image and upload it into Artifact Registry.

Make sure that the Artifact Registry API is enabled.

Run the following command to create a repository for Docker images.
```bash
gcloud artifacts repositories create vertex-pss-demo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Container image repository."
```

Run the following script to build and push the image to Artifact Registry.

```bash
bash scripts/build.sh
```

If you want to change the repository name and/or the location (region), make
sure to also change the AR_REPOSITORY AND REGION variables in the script
respectively.

Note that by default, the project ID and default region will be inferred
from your environment.

#### Configure training parameters

Make a copy of `pss_config.yaml.template` and save it as `pss_config.yaml`>
Make sure to update the following text:
- PROJECT: Your GCP project ID
- MODEL_BUCKET: Name of the GCS bucket for models
- DATASET_BUCKET: Name of the GCS bucket for datasets

If you changed the region where your Docker repository was created, make sure
to change that in the config file as well (default is `us-central1`).

Verify that the following fields are correct based on your environment.
- imageUri: Docker image tag of your trainer image.
- --model_dir: App specific-flag indicating the GCS path where model
    checkpoints will be stored.
- --train_pattern: App specific-flag indicating the GCS path where the training
    dataset will be read from.
- --val_pattern: App specific-flag indicating the GCS path where the validation
    dataset will be read from.

#### Run training on Vertex

After setting up your `pss_config.yaml` file, run the following script
to execute model training on Vertex AI:

```bash
bash scripts/vertex_train.sh
```
## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.
