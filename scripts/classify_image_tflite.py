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

"""Example using TF Lite to classify a given image using an Edge TPU.
   ```
   python3 classify_image_tflite.py \
     --export_dir exp_dir/export_dir \
     --model quant_edgetpu.tflite \
     --input images/apple_pie_0.jpg \
   ```
"""

import os
import argparse
import platform
import tensorflow as tf
from pathlib import Path
from typing import Union
from dishscan.datasets.nutrition5k import INGREDIENTS


# import tflite_runtime.interpreter as tflite
EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


def get_image_tensor(path_to_img: os.PathLike, input_specs):
    channels = input_specs["shape"][-1]
    img = tf.io.read_file(str(path_to_img))
    img = tf.image.decode_image(img, channels=channels)
    img = tf.image.resize(img, input_specs["shape"][1:-1], antialias=True)
    img = tf.reshape(img, input_specs["shape"])
    img = tf.cast(img, tf.dtypes.uint8)
    img = img.numpy()
    return img


def tflite_interpreter(
    model_file: os.PathLike,
    edge_tpu: bool = False,
) -> tf.lite.Interpreter:
    
    model_file = str(Path(args.model_file).resolve())
    delegates = None

    if edge_tpu:
        model_file, *device = model_file.split("@")
        delegates = [
            tf.lite.load_delegate(
                EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
            )
        ]
        interpreter = tf.lite.Interpreter(
            model_path=model_file, experimental_delegates=delegates
        )

    else:
        interpreter = tf.lite.Interpreter(
            model_path=model_file, experimental_delegates=delegates
        )

    return interpreter


def run_inference_tflite(
    interpreter: tf.lite.Interpreter,
    image: Union[tf.Tensor, os.PathLike],
    batch_size: int = 1,
) -> tf.Tensor:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_shape = tuple(input_details[0]["shape"][1:])
    interpreter.resize_tensor_input(
        input_details[0]["index"], (batch_size,) + image_shape
    )
    interpreter.allocate_tensors()
    img = (
        get_image_tensor(image, input_details[0])
        if isinstance(image, os.PathLike)
        else image
    )
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])
    return scores


def run_tflite(image: os.PathLike):
    edge_tpu = True if "edgetpu.tflite" in args.model_file else False
    interpreter = tflite_interpreter(args.model_file, edge_tpu=edge_tpu)
    scores = run_inference_tflite(interpreter, Path(image))
    return scores


# fmt:off
def configure() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # setup encoder from command-line
    parser.add_argument("-m","--model_file", type=str,
        help="model name i.e. a *.tflite to be used for inference.",)
    parser.add_argument("-i", "--input_file", required=True, type=str,
                         help="path to the input image to be inferred.")
    return parser.parse_args()
# fmt:on


def transform_scores_to_predictions(scores: tf.Tensor):
    predictions = tf.where(tf.sigmoid(scores.reshape(-1)) >= 0.5)
    predictions = predictions.numpy().flatten().tolist()
    predictions = map(lambda x: INGREDIENTS[x], predictions)
    return list(predictions)


if __name__ == "__main__":
    args = configure()
    input_file = str(Path(args.input_file).resolve())
    scores = run_tflite(input_file)
    predictions = transform_scores_to_predictions(scores)
    print(f'\npredicted ingredients:\n {predictions}')