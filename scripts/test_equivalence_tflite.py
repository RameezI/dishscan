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
from dishscan.datastreams import datastreams
from dishscan.utils import base_parser
from dishscan.metrics import AveragePrecisionScore, sigmoid_transform, AveragingMode


def tflite_interpreter(
    model_file: os.PathLike,
) -> tf.lite.Interpreter:
    delegates = None
    interpreter = tf.lite.Interpreter(
        model_path=model_file, experimental_delegates=delegates
    )
    return interpreter

def run_inference_tflite(
    interpreter: tf.lite.Interpreter,
    images: tf.Tensor,
) -> tf.Tensor:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(
        input_details[0]["index"], tf.shape(images)
    )
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], images)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])
    return scores

def convert_to_tflite(export_dir, name='model'):
    protocol = tf.saved_model.load(export_dir)
    concrete_function = protocol.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    _name = "{}_dynamic_int8_quant".format(name)
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    with open(Path(export_dir, f"{_name}.tflite"), "wb") as flat_buffer:
        flat_buffer.write(tflite_model)
    print("dynamic_int8 model generation finished!")
     
def test_equivalence(eval_stream: DataStream) -> None:
    run_dir = Path(args.logs_dir, args.name, args.run_id)
    export_dir = Path(run_dir, "export")

    protocol = tf.saved_model.load(export_dir)
    average_precision = AveragePrecisionScore(
        sigmoid_transform, averaging_mode=AveragingMode.MICRO, name="mAP"
    )

    # testing saved_model
    print('\n')
    average_precision.reset()
    for k, batch in enumerate(eval_stream):
        scores = protocol.predict_step_tf(batch["image"])
        average_precision.update(batch["ingredients"], scores)
        print(
            f"\rEvaluated {k+1}/{len(eval_stream)} batches using saved_model -- mAP: {average_precision.result()} ",
            end="",
        )
    print('\n')

    if args.create_tflite_model:
        convert_to_tflite(export_dir)

    model_file = str(Path(export_dir, 'model_dynamic_int8_quant.tflite').resolve())
    interpreter = tflite_interpreter(model_file)
    # testing int8 quantized (dynamic quantization) model
    average_precision.reset()
    for k, batch in enumerate(eval_stream):
        scores = run_inference_tflite(
            interpreter, batch["image"],
        )
        average_precision.update(batch["ingredients"], scores)
        print(
            f"\rEvaluated {k+1}/{len(eval_stream)} batches using tflite_dynamic_quant_8bit -- mAP: {average_precision.result()} ",
            end="",
        )
    print('\n')


def configuration() -> argparse.Namespace:
    """Returns the command line configuration"""
    parser = argparse.ArgumentParser(parents=[base_parser(training_component=False)])
    parser.add_argument(
        "--create_tflite_model",
        action="store_true",
        default=False,
        help="Weather to  generate a tflite (8-bit) quantized model file for the test",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = configuration()
    train_dl, eval_dl = datastreams(
        "nutrition5k",
        "/res448:1.*.*",
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        skip_normalization=True,
    )
    test_equivalence(eval_dl)