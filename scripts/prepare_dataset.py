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

"""Prepare and upload training dataset to Cloud Storage."""

import argparse
import logging
import os
import shlex
import subprocess

import tensorflow_datasets as tfds

logging.basicConfig(level=logging.INFO)


_CACHE_DIR = os.path.expanduser("~/tensorflow_datasets")


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="TFDS dataset.")
    parser.add_argument("gcs_output_dir", help="GCS output directory.")
    return parser.parse_args()


def main():
    """Main.

    Raises:
        ValueError: when GCS output directory is not valid.
    """
    args = _get_args()
    if not args.gcs_output_dir.startswith("gs://"):
        raise ValueError("Invalid GCS directory.")

    builder = tfds.builder(args.dataset)
    builder.download_and_prepare(
        download_dir=_CACHE_DIR,
        download_config=tfds.download.DownloadConfig(
            download_mode=tfds.download.GenerateMode.UPDATE_DATASET_INFO,
        ),
        file_format="tfrecord",
    )
    dataset_dir = os.path.join(_CACHE_DIR, args.dataset)
    command = f"gsutil -m cp -r {dataset_dir} {args.gcs_output_dir}"
    logging.info("Running command: %s", command)
    command_list = shlex.split(command)
    subprocess.run(command_list, check=True)


if __name__ == "__main__":
    main()
