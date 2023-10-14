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

from __future__ import annotations

import os
import sys
import re
import argparse
import platform
import json
import datetime
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import wraps
from dataclasses import dataclass
from typing import Union, Tuple
from enum import Enum
from typing import Any, Union, List
from typing import Union, List

# import tflite_runtime.interpreter as tflite
EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
]


def get_urid():
    now = datetime.datetime.now()
    urid = "{0:04d}{1:02d}{2:02d}-{3:02d}{4:02d}".format(
        now.year, now.month, now.day, now.hour, now.minute
    )
    return urid


@dataclass(frozen=True)
class ComputeComplexity:
    flops: int = np.nan
    trainable_parameters: int = np.nan
    non_trainable_parameters: int = np.nan

    @property
    def parameters(self):
        return self.trainable_parameters + self.non_trainable_parameters

    def __repr__(self):
        return (
            f"{'Total FLOPs':<20}: {self.flops:>20n}\n"
            f"{'Total Parameters':<20}: {self.parameters:>20n}\n"
            f"{'  Trainable':<20}: {self.trainable_parameters:>20n}\n"
            f"{'  Non-Trainable':<20}: {self.non_trainable_parameters:>20n}\n"
        )


class VerbosityLevel(Enum):
    KEEP_SILENT = "silent"
    UPDATE_AT_EPOCH = ("epoch",)
    UPDATE_AT_BATCH = "batch"


def serialize_to_json(self):
    def default_serializer(obj):
        _config = getattr(obj, "params", None)

        if _config is None:
            raise ValueError(
                "Serialization to JSON failed!\n"
                " The object does not have `params` attribute set, this is required attribute "
                "which must provide the minimum config to reconstruct the object. If you are\n"
                " serializing a custom class object, do not forgot to decorate your class "
                "`__init__` method with `capture_params` decorator, which can create `params`\n"
                " attribute for your object automatically. You can also supply the config"
                " manually by setting the `params` attribute of your object.\n"
                f" The concerned class is : `{type(obj).__name__}`"
            )

        _config.ClassificationMetricUpdater({"class_name": type(obj).__name__})
        return _config

    json_serialization = ""
    assert isinstance(
        self, object
    ), "The serialization candidate must be an instance of a class"
    try:
        json_serialization = json.dumps(
            self, default=default_serializer, sort_keys=True, indent=4
        )
    except ValueError as e:
        raise ValueError(
            "\n\nSerialization Failure!:\n"
            f"Failed while attempting to serialize an object of class `{type(self).__name__}`\n"
            f" {str(e)}"
        )

    return json_serialization


def capture_params(*args_outer, **kwargs_outer):
    ignore_list = kwargs_outer.get("ignore", [])
    apply_local_updates = kwargs_outer.get("apply_local_updates", False)

    # ignore_list.extend(['self'])

    def _capture_params(func):
        @wraps(_capture_params)
        def _wrapper_capture_params_(obj, *args, **kwargs):
            assert isinstance(
                obj, object
            ), "The capture params must be used on non-static methods of a class"
            _profile = sys.getprofile()

            parameters = [
                param
                for param in inspect.signature(func).parameters.values()
                if param.kind not in [inspect.Parameter.VAR_KEYWORD]
            ][1:]

            assert all(
                [
                    param.kind
                    not in [
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.VAR_POSITIONAL,
                    ]
                    for param in parameters
                ]
            ), (
                "The use of 'POSITIONAL_ONLY' and 'VAR_POSITIONAL' arguments"
                " is currently NOT supported by the capture function."
                " This feature is likely to be not supported in future as well."
                " Please use keywords only or positional arguments"
                "that support keywords instead."
            )

            obj.params = {
                param.name: param.default
                for param in parameters
                if param.name not in ignore_list
            }
            obj.params.update(
                {key: value for key, value in kwargs.items() if key not in ignore_list}
            )
            positional_params = list(inspect.signature(func).parameters.values())[
                1 : len(args) + 1
            ]
            obj.params.update(
                {param.name: value for param, value in zip(positional_params, args)}
            )

            def profiler(frame, event, _):
                if event == "return" and frame.f_back.f_code.co_name in [
                    "_wrapper_capture_params_"
                ]:
                    frame_locals = frame.f_locals
                    updates = {
                        key: value
                        for key, value in frame_locals.items()
                        if key in obj.params
                    }
                    obj.params.update(updates) if apply_local_updates else None

            try:
                sys.setprofile(profiler)
                func(obj, *args, **kwargs)
            finally:
                sys.setprofile(_profile)

        return _wrapper_capture_params_

    if args_outer and callable(args_outer[0]):
        return _capture_params(args_outer[0])
    else:
        return _capture_params


def get_image_tensor(path_to_img: os.PathLike, input_specs):
    channels = input_specs["shape"][-1]
    img = tf.io.read_file(str(path_to_img))
    img = tf.image.decode_image(img, channels=channels)
    img = tf.image.resize(img, input_specs["shape"][1:-1], antialias=True)
    img = tf.reshape(img, input_specs["shape"])
    img = tf.cast(img, tf.dtypes.uint8)
    img = img.numpy()
    return img


def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def make_spec_concrete(
    spec: Union[Tuple[str, tf.TensorSpec], tf.TensorSpec], concrete_value: int = 1
) -> Union[Tuple[str, tf.TensorSpec], tf.TensorSpec]:
    """
    Remove None values from the tensor spec and replace them with a specific value
    Args:
        spec:               The tensor spec with possible None dimensions
        concrete_value:     The replacement value for None dimensions
    Returns:
        TensorSpec with concrete values
    """
    if isinstance(spec, tuple):
        key, _spec = spec
        shape = _spec.shape
        if None in shape:
            shape = list(shape)
            shape[shape.index(None)] = concrete_value
            shape = tf.TensorShape(shape)
        concrete_spec = (key, tf.TensorSpec(shape, _spec.dtype))

    else:
        if not isinstance(spec, tf.TensorSpec):
            raise TypeError(
                "The input type is not supported!\n"
                "expected type: tf.TensorSpec\n"
                f"received: {type(spec).__name}"
            )
        shape = spec.shape
        if None in shape:
            shape = list(shape)
            shape[shape.index(None)] = concrete_value
            shape = tf.TensorShape(shape)
        concrete_spec = tf.TensorSpec(shape, spec.dtype)

    return concrete_spec


def camel_to_snake(name: str):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def snake_to_camel(name: str, splitter="_"):
    components = name.split(splitter)
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


class KeyValueAction(argparse.Action):
    def __call__(self, _, namespace, values, option_string=None):
        if option_string is not None:
            raise ValueError
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            if value.lower() in ["true", "yes", "no", "false"]:
                value = True if value.lower() in ["true", "yes"] else False
            else:
                value = float(value) if is_float(value) else value
            getattr(namespace, self.dest)[key] = value

    # fmt:off
def base_parser(training_component=True, data_component=True,
) -> argparse.ArgumentParser:


    verbose_choices = [e.value for e in VerbosityLevel]
    logging_choices = ['debug', 'info', 'warning', 'error', 'critical']                
    parser = argparse.ArgumentParser(description="command line arguments for the dishscan program",
                                     add_help=False)
    
    # Using these standard args make it easier to write consistent launch files.
    parser.add_argument('--logs_dir', required=False, type=str,
                        default=str(Path(Path.home(), 'tensorflow_logs')),
                        help='path to the top level tensorflow_log directory')
    parser.add_argument('--name', required=False, type=str, default='dishscan',
                        help='name of the training job')
    parser.add_argument('--run_id', required=False, type=str, default=None,
                        help='run_id/version for the job')
    
    if data_component:
        parser.add_argument('--datasets_dir', required=False, type=str,
                            default=str(Path(Path.home(), 'tensorflow_datasets')),
                            help='path to the directory that contains tensorflow datasets')
        parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help='The batch size for all dataloaders')
        parser.add_argument('--skip_normalization', action='store_true', default=False,
                        help='Weather to skip data normalization transform while feeding the data.')

    if training_component:
        parser.add_argument('--max_epochs', required=False, type=int, default=50,
                            help='The maximum number of allowed epochs for training/fine-tuning jobs')
        parser.add_argument('--lr', required=False, type=float, default=1e-3,
                            help='The base learning rate for the training jobs.')
        parser.add_argument('--run_eagerly', action='store_true', default=False,
                            help='Weather to execute the training protocols eagerly or in graph mode')
        parser.add_argument('--verbose', required=False, default='batch',
                            type=VerbosityLevel, choices=verbose_choices,
                            help='verbosity level of the Dishscan trainer')
        parser.add_argument('--logging_level', required=False, default='warning',
                            type=str.lower, choices=logging_choices,
                            help='log level of the Dishscan trainer')
        # setup encoder from command-line
        parser.add_argument('--encoder_module', required=False, type=str, default='encoders',
                            help='The module where the encoder class is searched')
        parser.add_argument('--encoder', required=False, type=str, default='MobileNetV1')
        parser.add_argument('--encoder_config', nargs='*', action=KeyValueAction, default=dict())

        # setup decoder from command line
        parser.add_argument('--decoder_module', required=False, type=str, default='decoders',
                            help='The module where the decoder class is searched')
        parser.add_argument('--decoder', required=False, type=str, default='GlobalAvgPoolDecoder')
        parser.add_argument('--decoder_config', nargs='*', action=KeyValueAction, default=dict())

        parser.add_argument('--skip_training', action='store_true', default=False,
                            help='Weather to skip the training process altogether.')
    
    return parser