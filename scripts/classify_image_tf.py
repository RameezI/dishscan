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
from dishscan.datasets.nutrition5k import INGREDIENTS

def get_image_tensor(path_to_img: os.PathLike):
    img = tf.io.read_file(str(path_to_img))
    img = tf.image.decode_image(img)
    return img


def tf_interpreter(export_dir: str) -> tf.Module:
    interpreter = tf.saved_model.load(export_dir)
    return interpreter


def run_inference_tf(interpreter: tf.Module, image: tf.Tensor) -> tf.Tensor:
    image = tf.reshape(image, (1, 448, 448, 3))
    scores = interpreter.predict_step_tf(image)
    return scores.numpy()

def transform_scores_to_predictions(scores: tf.Tensor):
    predictions = tf.where(tf.sigmoid(scores.reshape(-1)) >= 0.5)
    predictions = predictions.numpy().flatten().tolist()
    predictions = map(lambda x: INGREDIENTS[x], predictions)
    return list(predictions)


def configure() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # setup encoder from command-line
    parser.add_argument(
        "-d",
        "--export_dir",
        required=True,
        type=str,
        help="path to the exported models, this is the directory"
        " that contains the saved_model.pb file",
    )
    parser.add_argument(
        "-i", "--input_file", required=True, type=str,
          help="path to the input image"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = configure()
    image = get_image_tensor(args.input_file)
    interpreter = tf_interpreter(args.export_dir)
    scores = run_inference_tf(interpreter, image)
    predictions = transform_scores_to_predictions(scores)
    print(f'\npredicted ingredients:\n {predictions}')