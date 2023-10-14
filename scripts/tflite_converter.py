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

import os
import argparse
import tensorflow as tf
from pathlib import Path
from dishscan import DataStream
from dishscan.utils import base_parser
from dishscan.datastreams import datastreams
from typing import List


def convert_to_tflite(
    export_dir: os.PathLike,
    quantization_modes: List[str] = None,
    name="model",
    representative_dataset: DataStream = None,
):
    def representative_data_generator():
        for batch in representative_dataset:
            yield [batch["image"]]

    valid_quantization_modes = ("DYNAMIC_INT8", "FIXED_INT8", "DYNAMIC_FLOAT16")
    if quantization_modes is None:
        quantization_modes = valid_quantization_modes
    else:
        assert quantization_modes in valid_quantization_modes

    protocol = tf.saved_model.load(export_dir)
    concrete_function = protocol.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]

    if "DYNAMIC_FLOAT16" in quantization_modes:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        _name = "{}_dynamic_float16_quant".format(name)
        converter.post_training_quantize = True
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(
            os.path.join(export_dir, "{}.tflite".format(_name)), "wb"
        ) as flat_buffer:
            flat_buffer.write(tflite_model)
        print("dynamic_float16 quantization finished!")

    if "DYNAMIC_INT8" in quantization_modes:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        _name = "{}_dynamic_int8_quant".format(name)
        converter.post_training_quantize = True
        tflite_model = converter.convert()
        with open(Path(export_dir, f"{_name}.tflite"), "wb") as flat_buffer:
            flat_buffer.write(tflite_model)
        print("dynamic_int8 quantization finished!")

    if "FIXED_INT8" in quantization_modes:
        if representative_dataset is None:
            raise RuntimeError(
                "Request to quantize the model with `FIXED_INT8` quantization, but no representative dataset provided.\n\
             Quitting!\n"
            )

        # create converter
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        _name = "{}_fixed_int8_quant".format(name)  # converted_model_name
        # convert the model with the fixed ranges (based on the representative dataset)
        converter.post_training_quantize = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            representative_data_generator
        )
        tflite_model = converter.convert()
        with open(
            os.path.join(export_dir, "{}.tflite".format(_name)), "wb"
        ) as flat_buffer:
            flat_buffer.write(tflite_model)
        print("dynamic_int8 quantization finished!")


def configuration() -> argparse.Namespace:
    """Returns the command line configuration"""
    """Returns the command line configuration"""
    parser = argparse.ArgumentParser(
        parents=[base_parser(training_component=False)]
    )
    # setup default base arguments
    parser.set_defaults(datasets_dir=str(Path(Path.home(), "tensorflow_datasets")))
    parser.set_defaults(logs_dir=str(Path(Path.home(), "tensorflow_logs")))
    parser.set_defaults(batch_size=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = configuration()
    run_dir = Path(args.logs_dir, args.name, args.run_id)
    export_dir = Path(run_dir, "export")

    train_dl, _ = datastreams(
        "nutrition5k",
        "/res448:1.*.*",
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        skip_normalization=True,
    )
    convert_to_tflite(export_dir, representative_dataset=train_dl)