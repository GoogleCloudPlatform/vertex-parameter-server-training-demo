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

"""Run model training or start distributed training server."""

import argparse
import json
import logging
import multiprocessing
import os
import sys

import portpicker
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

_IMAGE_CHANNELS = 3


# pylint: disable=no-member


def _get_args():
    """Returns parsed command line arguments."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # startserver command
    server_parser = subparsers.add_parser(
        "startserver", help="Start TensorFlow Distributed Server"
    )
    server_parser.add_argument(
        "--task_type",
        default=None,
        type=str,
        help="Override task type for distributed server",
    )
    server_parser.add_argument(
        "--task_index",
        default=None,
        type=int,
        help="Override task index for either worker or ps.",
    )

    # train command
    train_parser = subparsers.add_parser(
        "train", help="Run distributed model training."
    )
    train_parser.add_argument("--model_dir", help="Output model directory.")
    train_parser.add_argument("--train_pattern", help="Training dataset pattern.")
    train_parser.add_argument("--val_pattern", help="Validation dataset pattern.")
    train_parser.add_argument("--batch_size", type=int, help="Per-replica batch size.")
    train_parser.add_argument(
        "--train_steps", type=int, help="Number of training steps."
    )
    train_parser.add_argument(
        "--val_steps", type=int, help="Number of validation steps."
    )
    train_parser.add_argument("--learning_rate", type=float, help="learning rate")
    train_parser.add_argument("--momentum", type=float, help="SGD momentum value")
    train_parser.add_argument(
        "--units", type=int, default=32, help="number of units in last hidden layer"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )
    train_parser.add_argument(
        "--strategy",
        default=None,
        help="Distributed training strategy.",
    )
    train_parser.add_argument(
        "--use_in_process_cluster",
        default=False,
        action="store_true",
    )

    # train ... in_process command
    train_subparsers = train_parser.add_subparsers(
        dest="train_subcommand", help="Train subcommands."
    )
    in_process_parser = train_subparsers.add_parser(
        "in_process",
        help=(
            "Use in-process cluster."
            " Valid only when train_strategy is 'ParameterServerStrategy'."
        ),
    )
    in_process_parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers."
    )
    in_process_parser.add_argument(
        "--num_ps", type=int, default=0, help="Number of parameter servers."
    )

    args = parser.parse_args()

    return args


def _preprocess_data(example: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocesses data.

    Args:
        example: Example dict mapping feature names to tensors.

    Returns:
        Tuple of preprocessed image tensor and label
    """
    image = tf.image.resize(example["image"], (150, 150))
    image = tf.cast(image, tf.float32) / 255.0
    return image, example["label"]


def _parse_example(example_proto: str) -> dict[str, tf.Tensor]:
    """Returns parsed TF example proto."""
    example = tf.io.parse_single_example(example_proto, _FEATURE_SPEC)
    example["image"] = tf.io.decode_png(example["image"], channels=_IMAGE_CHANNELS)
    return example


def _load_dataset(
    file_pattern: str,
    batch_size: int,
    shuffle_buffer_size: int = 1000,
) -> tf.data.TFRecordDataset:
    """Loads a TFRecord dataset from file/s."""
    files = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_preprocess_data)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat()
    return dataset


def _create_model(units, learning_rate, momentum):
    """Defines and compiles model."""

    inputs = tf.keras.Input(shape=(150, 150, _IMAGE_CHANNELS))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum
        ),
        metrics=["accuracy"],
    )
    return model


def _startserver(task_type: str | None = None, task_index: int | None = None):
    """Starts a TF distributed server."""
    if "TF_CONFIG" not in os.environ:
        raise ValueError("TF_CONFIG environment variable must be set")

    if task_type or (task_index is not None and task_index >= 0):
        tf_config = json.loads(os.environ["TF_CONFIG"])
        if task_type is not None:
            tf_config["task"]["type"] = task_type
        if task_index is not None:
            tf_config["task"]["index"] = task_index
        os.environ["TF_CONFIG"] = json.dumps(tf_config)

    logging.info("Using TF_CONFIG=%s", os.environ["TF_CONFIG"])

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in {"worker", "ps"}:
        logging.info(
            "Starting server for task='%s', index='%s",
            cluster_resolver.task_type,
            cluster_resolver.task_id,
        )
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True,
        )
        server.join()
    else:
        raise ValueError(f"Unsupported task type: {cluster_resolver.task_type}")


def _get_strategy(
    strategy_name: str | None = None,
    cluster_resolver: tf.distribute.cluster_resolver.ClusterResolver | None = None,
) -> tf.distribute.Strategy:
    """Returns Strategy object from strategy name."""
    strategy = None
    if strategy_name == "ParameterServerStrategy":
        if not cluster_resolver:
            cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)
    elif strategy_name == "" or strategy_name is None:
        strategy = tf.distribute.get_strategy()
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    return strategy


def _train(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    model_dir: str,
    train_pattern: str,
    val_pattern: str,
    batch_size: int,
    train_steps: int,
    val_steps: int,
    learning_rate: float,
    momentum: float,
    units: int,
    epochs: int,
    train_strategy: str,
    cluster_resolver: tf.distribute.cluster_resolver.ClusterResolver | None = None,
) -> None:
    """Runs model training.

    Args:
        model_dir: Directory where the trained model will be saved.
        train_pattern: File pattern (e.g., glob) for training data.
        val_pattern: File pattern (e.g., glob) for validation data.
        batch_size: Number of samples per training/validation step per worker.
        train_steps: Total number of training steps per epoch.
        val_steps: Total number of validation steps per epoch.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum value for the optimizer.
        units: Number of units in the hidden layer.
        epochs: Total number of training epochs.
        train_strategy: Distributed training strategy.
        cluster_resolver: Resolver for getting training cluster info.
    """
    # Create distributed strategy
    strategy = _get_strategy(train_strategy, cluster_resolver)
    logging.info("Using strategy: %s", strategy)

    # Load datasets
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    train_data = _load_dataset(train_pattern, global_batch_size)
    validation_data = _load_dataset(val_pattern, batch_size)

    # Wrap model variables within scope
    with strategy.scope():
        model = _create_model(units, learning_rate, momentum)

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "ckpt/checkpoint-model-keras"),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    # Train model
    model.fit(
        train_data,
        validation_data=validation_data,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
    )


def _create_in_process_cluster(
    *,
    num_workers: int = 1,
    num_ps: int = 0,
) -> tf.distribute.cluster_resolver.ClusterResolver:
    """Creates an in-process distributed training cluster.

    Reference:
    https://www.tensorflow.org/tutorials/distribute/parameter_server_training

    Args:
        num_workers: Number of workers.
        num_ps: Number of parameter servers.

    Returns:
        Cluster resolver object.
    """
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

    cluster_dict = {}
    cluster_dict["worker"] = [f"localhost:{port}" for port in worker_ports]
    if num_ps > 0:
        cluster_dict["ps"] = [f"localhost:{port}" for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    for i in range(num_workers):
        tf.distribute.Server(
            cluster_spec,
            job_name="worker",
            task_index=i,
            config=worker_config,
            protocol="grpc",
        )

    for i in range(num_ps):
        tf.distribute.Server(cluster_spec, job_name="ps", task_index=i, protocol="grpc")

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc"
    )
    return cluster_resolver


def main():
    """Main."""
    args = _get_args()

    if args.command == "startserver":
        try:
            _startserver(args.task_type, args.task_index)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Start server failed: %s", e)

    elif args.command == "train":
        if args.train_subcommand == "in_process":
            cluster_resolver = _create_in_process_cluster(
                num_workers=args.num_workers, num_ps=args.num_ps
            )
        else:
            cluster_resolver = None
        try:
            _train(
                model_dir=args.model_dir,
                train_pattern=args.train_pattern,
                val_pattern=args.val_pattern,
                batch_size=args.batch_size,
                train_steps=args.train_steps,
                val_steps=args.val_steps,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                units=args.units,
                epochs=args.epochs,
                train_strategy=args.strategy,
                cluster_resolver=cluster_resolver,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Model training failed: %s", e)

    else:
        logging.error("Unsupported command: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
