# Copyright (c) 2023 Rameez Ismail
#
# Licensed under the The MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#    https://opensource.org/license/mit/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Author(s): Rameez Ismail
# Email(s):  rameez.ismail@protonmail.com

import json
import argparse
import inspect
import importlib
import tensorflow as tf
from pathlib import Path
from dishscan import Trainer
from dishscan import DataStream
from dishscan import datastreams
from dishscan import MLCProtocol
from dishscan.utils import base_parser


def get_instantiated_models():
    encoder_module = importlib.import_module(
        f".models.{args.encoder_module}", package="dishscan"
    )
    decoder_module = importlib.import_module(
        f".models.{args.decoder_module}", package="dishscan"
    )
    # encoder-
    class_members = inspect.getmembers(encoder_module, inspect.isclass)
    class_members = dict(
        filter(lambda item: issubclass(item[1], tf.keras.Model), class_members)
    )
    encoder: type = class_members[args.encoder]

    # decoder
    class_members = inspect.getmembers(decoder_module, inspect.isclass)
    class_members = dict(
        filter(lambda item: issubclass(item[1], tf.keras.Model), class_members)
    )
    decoder: type = class_members[args.decoder]

    models = {
        "encoder": encoder(**args.encoder_config),
        "decoder": decoder(**args.decoder_config),
    }
    return models


def evaluate(eval_stream: DataStream) -> None:
    models = get_instantiated_models()
    epoch = tf.Variable(0, dtype=tf.int64)

    run_dir = Path(args.logs_dir, args.name, args.run_id)
    ckpt_dir = Path(run_dir, "checkpoints")

    # restore from checkpoint
    checkpoint = tf.train.Checkpoint(epoch=epoch, **models)
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    checkpoint.restore(latest_ckpt)

    # Define a training/evaluation/prediction protocol
    protocol = MLCProtocol(("image", "ingredients"))
    protocol.compile(models)

    for k, batch in enumerate(eval_stream):
        protocol.evaluate_step_tf(batch)
        print(f"\rEvaluated {k+1}/{len(eval_stream)} batches ... ", end="")

    metrics = {
        metric.name: float(metric.result().numpy()) for metric in protocol.metrics
    }
    print(json.dumps(metrics, indent=4))
    with open(Path(run_dir, "evaluation.json"), "w") as fp:
        json.dump(metrics, fp)


def train(train_stream: DataStream, val_stream: DataStream):
    # Trainer
    serve_signature = tf.TensorSpec((None, 448, 448, 3), dtype=tf.uint8)
    trainer = Trainer(
        train_stream,
        eval_stream=val_stream,
        logs_dir=args.logs_dir,
        name=args.name,
        run_id=args.run_id,
        ckpt_opts={"monitor": "mAP/val", "mode": "max"},
        export_opts={"signature": serve_signature},
    )

    # Add constituents
    trainer.push_model(
        args.encoder_module,
        alias="encoder",
        inception_class=args.encoder,
        config=args.encoder_config,
    )

    trainer.push_model(
        args.decoder_module,
        alias="decoder",
        inception_class=args.decoder,
        config=args.decoder_config,
    )

    # Define a training/evaluation/prediction protocol
    MLCProtocol.ITERATIONS_ESTIMATE = len(train_stream) * args.max_epochs
    MLCProtocol.INITIAL_LR = args.lr
    protocol = MLCProtocol(("image", "ingredients"))

    # Run the trainer as per the protocol/s
    trainer.spin(protocol, max_epochs=args.max_epochs, run_eagerly=args.run_eagerly)


def configuration() -> argparse.Namespace:
    """Returns the command line configuration"""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    return parser.parse_args()


if __name__ == "__main__":
    args = configuration()
    train_dl, eval_dl = datastreams(
        "nutrition5k",
        "/res448:1.*.*",
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        skip_normalization=args.skip_normalization,
    )
    train(train_dl, eval_dl) if not args.skip_training else None
    evaluate(eval_dl)