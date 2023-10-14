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

"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import json
import inspect
import contextlib
import importlib
import locale
import uuid
import logging
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import Union, List, Tuple, Dict, Iterable, Type, Any

# graph analysis tools
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from .callbacks import Callback
from .datastreams import DataStream
from .protocol import Protocol

from .callbacks import CheckPointer, ProgressParaphraser, Exporter
from .progressbar import ProgressBar, ModeProgressBar

from .utils import (
    get_urid,
    make_spec_concrete,
    ComputeComplexity,
    VerbosityLevel,
)

# global settings
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)  # put 10 for high verbosity
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

logger = logging.getLogger(__name__)

# Aliases
CallbackList = tf.keras.callbacks.CallbackList


class Trainer:
    # TensorFlow Distribution Schemes Validated by the Trainer
    named_distribution_schemes = {
        "one_device_cpu": tf.distribute.OneDeviceStrategy(device="/cpu:0"),
        "one_device_gpu": tf.distribute.OneDeviceStrategy(device="/gpu:0"),
        "mirrored": tf.distribute.MirroredStrategy(),
    }

    def __init__(
        self,
        train_stream: Union[DataStream, tf.data.Dataset],
        eval_stream: Union[None, DataStream, tf.data.Dataset] = None,
        logs_dir: Union[None, str] = None,
        name: Union[str, None] = None,
        run_id: Union[None, str] = None,
        distributor: Union[None, tf.distribute.Strategy] = None,
        ckpt_opts: Union[None, Dict[str, Any]] = None,
        export_opts: Union[None, Dict[str, Any]] = None,
    ):
        """
        Args:
            train_stream:
            eval_stream:
            name:
            run_id:
            logs_dir:
            distributor:
            ckpt_opts:          Checkpoint options for the trainer. User can supply a dictionary of desired options.
        """

        default_log_dir = os.path.join(os.path.expanduser("~"), "tensorflow_logs")
        self._logs_dir = default_log_dir if logs_dir is None else logs_dir
        self._name = name
        self._run_id = get_urid() if run_id is None else run_id
        self._models = {}

        # convert to canonical form
        if isinstance(train_stream, tf.data.Dataset):
            train_stream = DataStream(train_stream, split="train")

        if isinstance(eval_stream, tf.data.Dataset):
            eval_stream = DataStream(eval_stream, split="test")

        # examples count
        self._train_samples_count = np.nan
        self._eval_samples_count = np.nan
        self._data_streams = self._connect_streams(train_stream, eval_stream)
        self._protocol: Union[None, Protocol] = None
        self._epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.int64)

        try:
            self._distributor_scope = (
                distributor.scope()
                if distributor is not None
                else contextlib.suppress()
            )
        except Exception as e:
            logger.error("\n\nFailed to create the requested distribution context!\n")
            raise

        self._checkpointer_config: Union[Dict[str, Any], None] = None
        self._export_config: Union[Dict[str, Any], None] = None
        self.predict_complexity: Union[None, ComputeComplexity] = None

        self.checkpointer_config = ckpt_opts
        self.export_config = export_opts

        # Tensorboard Summaries
        self._train_summarizer: Union[None, tf.summary.SummaryWriter] = None
        self._val_summarizer: Union[None, tf.summary.SummaryWriter] = None

        # Progress bar
        self._progress_bar = ProgressBar()

    @property
    def logs_dir(self):
        return self._logs_dir

    @property
    def name(self) -> str:
        return str(self)

    @property
    def run_id(self):
        return self._run_id

    @logs_dir.setter
    def logs_dir(self, value):
        self._logs_dir = value

    @name.setter
    def name(self, value):
        self._name = value

    @run_id.setter
    def run_id(self, value: str):
        self._run_id = value

    @property
    def run_dir(self):
        run_dir = None
        if self.name is not None:
            run_dir = Path(self.logs_dir, self.name, self.run_id)
        return run_dir

    @property
    def checkpoint_dir(self) -> os.PathLike:
        checkpoint_dir = None
        if self.run_dir is not None:
            checkpoint_dir = Path(self.run_dir, "checkpoints")
        return checkpoint_dir

    @property
    def epoch(self) -> int:
        return self._epoch.value().numpy()

    @property
    def tensorboard_dir(self) -> str:
        tb_summaries_dir = None
        if self.run_dir is not None:
            tb_summaries_dir = os.path.join(self.run_dir, "summaries")
        return tb_summaries_dir

    @property
    def summary_writers(self):
        uid = str(uuid.uuid4())[:8]
        if self._train_summarizer is None:
            self._train_summarizer = tf.summary.create_file_writer(
                str(Path(self.tensorboard_dir, "train")), name=f"spin_{uid}"
            )
        if self._val_summarizer is None:
            self._val_summarizer = tf.summary.create_file_writer(
                str(Path(self.tensorboard_dir, "val")), name=f"spin_{uid}"
            )
        return {"train": self._train_summarizer, "test": self._val_summarizer}

    @property
    def models(self):
        keys = list(self._models.keys())
        models = None if not keys else self._models
        return models

    # Configure the underlying Checkpointer Callback
    @staticmethod
    def _configure_ckpt(user_config: Union[None, dict] = None):
        user_config = {} if user_config is None else user_config
        ckpt_config = {
            "monitor": "loss/val",
            "silent": False,
            "keep_best": True,
            "max_to_keep": 3,
            "mode": "auto",
        }
        ckpt_config.update(user_config)
        return ckpt_config

    # Configure the underlying Checkpointer Callback
    @staticmethod
    def _configure_export(user_config: Union[None, dict] = None):
        user_config = {} if user_config is None else user_config
        export_config = {
            "signature": (),
        }
        export_config.update(user_config)
        return export_config

    @property
    def checkpointer_config(self) -> Dict:
        if self._checkpointer_config is None:
            self._checkpointer_config: Dict[str, Any] = self._configure_ckpt(None)
        return self._checkpointer_config

    @checkpointer_config.setter
    def checkpointer_config(self, value: Union[Dict, None]):
        self._checkpointer_config = self._configure_ckpt(value)

    @property
    def export_config(self) -> Dict:
        if self._export_config is None:
            self._export_config: Dict[str, Any] = self._configure_export(None)
        return self._export_config

    @export_config.setter
    def export_config(self, value: Union[Dict, None]):
        self._export_config = self._configure_export(value)

    # Connect Data Streams with the trainer
    def _connect_streams(
        self, train: DataStream, evaluate: DataStream = None
    ) -> Dict[str, DataStream]:
        """
        This Connects the data_streams with the trainer. The data streams are mutable objects, therefore,  any changes
        to data streams in the outer scope will be reflected within the trainer as well and vice-versa.
        Args:
            train: The 'DataSteam' object that will be used for the training purpose
            evaluate: The 'DataSteam' object that will be used for the validation purpose
        Returns: None
        """
        self._train_samples_count = train.examples_count
        self._eval_samples_count = (
            evaluate.examples_count if evaluate is not None else np.nan
        )
        return {"train": train, "evaluate": evaluate}

    # Add  a  Model
    @staticmethod
    def assert_inception_sanity(matched_keys, search_term, module_name):
        no_match_msg = (
            f"\n Unable to match an inception_class in the `{module_name}` module:\n"
            f" In case there are multiple models (tf.keras.Model) in the module, the top level"
            f" class, i.e. the inception class, must have the same name\n"
            f" as the module itself (excluding the underscores, case-insensitive)."
            f" When a different name is desired, please provide it explicitly via\n"
            f" the `inception_class` keyword argument."
        )

        multi_match_msg = (
            f"\nUnable to determine a unique inception_class:\n"
            f"Multiple top level classes identified: {matched_keys}\n"
            f"A possible failure scenario is the use of the case-sensitive class"
            f" names in the module, if the error persists,\nplease provide the name"
            f" of the inception class explicitly via the `inception_class` keyword argument."
        )

        if not matched_keys:
            raise RuntimeError(no_match_msg)

        if len(matched_keys) > 1:
            raise RuntimeError(multi_match_msg)

    def push_model(
        self,
        model: Tuple[str, Type[tf.keras.Model]],
        config: Union[dict, None] = None,
        alias: Union[None, str] = None,
        pkg: Union[None, str] = None,
        inception_class: Union[None, str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        This pushes a model to the trainer. Multiple models can be added by repeated calls to the method, in which case
        a dictionary of models is maintained by the trainer. The trainer forwards these models to any strategy object
        passed to the spin method of the trainer object.

        Args:
            model:              A type derived from tf.keras.Model class or a string referring to the module that
                                implements the subclass
            config:             Configuration required to instantiate the `model`.
            alias:              The key under which the model will be saved, the model will be renamed accordingly
            pkg:                Python package containing the module (ignored when a tf.kera.Model subclass is provided)
            inception_class:    The top level class of the module that is to be instantiated. When None the trainer
                                determines the class itself.
            overwrite:          When True, trainer overwrites any existing model with the same alias.

        Returns:                None
        """

        # create run_dir if it does not exit
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with self._distributor_scope:
            config = {} if config is None else config

            def get_unique_key(models: dict, candidate_key: str):
                for post_fix in range(1, len(models) + 1):
                    if candidate_key not in models.keys():
                        break
                    else:
                        candidate_key = f"{candidate_key}_{post_fix}"
                return candidate_key

            if isinstance(model, str):
                pkg = "dishscan" if pkg is None else pkg
                module = importlib.import_module(f".models.{model}", package=pkg)
                search_term = model if inception_class is None else inception_class
                class_members = inspect.getmembers(module, inspect.isclass)
                class_members = dict(
                    filter(
                        lambda item: issubclass(item[1], tf.keras.Model), class_members
                    )
                )

                if inception_class is None:
                    matched_keys = [
                        key
                        for key in class_members.keys()
                        if key.lower() == search_term.replace("_", "").lower()
                        or len(class_members) == 1
                    ]
                else:
                    matched_keys = [
                        key for key in class_members.keys() if key == search_term
                    ]

                self.assert_inception_sanity(matched_keys, search_term, model)
                inception_class = class_members[matched_keys[0]]
            else:
                inception_class = model

            if not (
                isinstance(inception_class, type(tf.keras.Model))
                and issubclass(inception_class, tf.keras.Model)
            ):
                raise TypeError(
                    "The model cannot be added due to type mismatch, please"
                    " provide a class derived from `tf.keras.Model`.\n"
                    f"Expected: {tf.keras.Model}\n"
                    f"Received: {inception_class}\n"
                )

            model = inception_class(**config)
            default_alias = (
                get_unique_key(self._models, "model") if not overwrite else "model"
            )
            alias = default_alias if alias is None else alias

            if not overwrite:
                assert alias not in self._models.keys(), (
                    f"Alias reuse!\n"
                    f"The model with alias=`{alias}` already exists. Use `force` option if an overwrite is indented"
                )

            if model.built:
                logger.warning(
                    """ The model name can not be changed recursively! The model is already built and
                 weights are assigned names at the inception phase. The weights name are not consistent with the
                 provided alias, which could lead to errors and confusion!"""
                )
            model._name = alias
            self._models.update({alias: model})

    def _compile_protocol(self, protocol: Protocol):
        with self._distributor_scope:
            if not isinstance(protocol, Protocol):
                raise TypeError(
                    f"""The strategy must be a subclass of {type(Protocol)}
                                Expected: {Protocol}
                                Received: {protocol}"""
                )
            models_dict = self._models
            if not models_dict:
                raise RuntimeError(
                    f"The trainer has no computational graph, unable to compile the strategy.\n"
                    f"You can add one or more models to the trainer via add_model method:\n"
                    f"{self.push_model.__doc__}"
                )
            try:
                protocol.compile(self._models)

            except Exception as e:
                logger.error(str(e))
                logger.error("The strategy compilation failed!")
                raise e
        return True

    def configure_callbacks(
        self, callback_list: List[Callback], verbose: VerbosityLevel
    ):
        """
        Drop specific default callbacks if relevant callback is found in user callbacks.
        Otherwise, return a default callback list.
        Args:
            callback_list:      A list of user supplied callbacks
            verbose:            The verbosity of the progress reporting callback
        Returns:
        """
        if callback_list:
            logger.warning(
                "\n\nExternal supply of a callback list is not yet"
                " fully tested, please make sure your callback does"
                " not conflict with the defaults "
            )

        callback_list.append(
            CheckPointer(
                self.checkpoint_dir,
                **self.checkpointer_config,
                progress_bar=self._progress_bar,
            )
        )
        callback_list.append(Exporter(self.checkpoint_dir, **self._export_config))
        callback_list.append(ProgressParaphraser(self._progress_bar, verbose=verbose))

        for cb in callback_list:
            cb.run_dir = self.run_dir
            cb.protocol = self._protocol
            cb.writers = self.summary_writers
            cb.models = self.models

        return callback_list

    def _load_checkpoint(self, checkpoint, silent=True):
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        status = checkpoint.restore(latest_ckpt)
        print(f"Restored from: \n{latest_ckpt} \n") if not silent else None
        return status

    def spin(
        self,
        protocol: Protocol,
        max_epochs: int = 100,
        callback_list: Union[None, List[Callback]] = None,
        warm_start: bool = True,
        run_eagerly: bool = False,
        verbose: VerbosityLevel = VerbosityLevel.UPDATE_AT_BATCH,
        **kwargs,
    ):
        """
        When spin method is invoked: it compiles the protocol executes the training/evaluation tasks
        as per the given protocol.

        Args:
            protocol:
            max_epochs:
            callback_list:
            warm_start:
            run_eagerly:
            serving_signature:
            verbose:
            **kwargs:

        Returns:

        """

        with self._distributor_scope:
            self._compile_protocol(protocol)

        if max_epochs < 1:
            raise ValueError(
                "Invalid value for `max_epochs`."
                "The `max_epochs` must be a positive integer.\n"
                f"max_epoch == {max_epochs}"
            )

        if not protocol.compiled:
            raise RuntimeError(
                f"The training/evaluation protocol is not in place yet!"
                f"It might be that compilation was unsuccessful."
            )

        self._epoch.assign(0)
        self._protocol = protocol

        assert self.name, self.run_id
        callback_list = [] if callback_list is None else callback_list
        callback_list = self.configure_callbacks(callback_list, verbose)
        callbacks = CallbackList(callback_list)

        do_validation = False if self._data_streams["evaluate"] is None else True
        eval_stream = (
            self._data_streams["evaluate"]
            if do_validation
            else self._data_streams["train"]
        )

        self.write_config_to_disk()
        print(json.dumps(self.to_json(), indent=4)) if verbose not in [
            VerbosityLevel.KEEP_SILENT
        ] else None

        try:
            # Profile the underlying predict method to estimate Flops & Parameters
            silent = verbose == VerbosityLevel.KEEP_SILENT
            self._profile_inference_graph(eval_stream, silent=silent)
        except NotImplementedError:
            logger.warning(
                f"\n{'Lacking a predict step':-^35}\n"
                "The strategy does not provide a `call` method,"
                " skipping inference graph analysis!\n"
            )

        # Attempt a warm restart if checkpoint exists and warm_start is set to true
        if warm_start and os.path.isfile(Path(self.checkpoint_dir, "checkpoint")):
            checkpoint = tf.train.Checkpoint(
                epoch=self._epoch, **self._models, objectives=self._protocol.objectives
            )
            try:
                silent = verbose == VerbosityLevel.KEEP_SILENT
                ckpt_load_status = self._load_checkpoint(checkpoint, silent=silent)
                ckpt_load_status.assert_existing_objects_matched()
            except AssertionError as e:
                self._epoch.assign(0)
                logging.error(str(e))
                raise AssertionError(
                    "Checkpoint load failure!\n"
                    f"checkpoint directory: {self.checkpoint_dir}"
                )

        # Initial evaluation
        if not kwargs.pop("skip_initial_evaluation", False):
            self._progress_bar.mode = ModeProgressBar.EVAL_ONLY
            split_name = eval_stream.split_name
            print(f"Evaluation on the `{split_name}` split:") if verbose not in [
                VerbosityLevel.KEEP_SILENT
            ] else None

            samples_init_eval = (
                self._eval_samples_count if do_validation else self._train_samples_count
            )
            callbacks.on_test_begin(logs={"samples/eval": samples_init_eval})
            logs_init_eval = protocol.evaluate(
                eval_stream, callbacks, run_eagerly=run_eagerly
            )
        else:
            logs_init_eval = {}
        #
        callbacks.on_test_end(logs_init_eval)
        self._progress_bar.mode = ModeProgressBar.DEFAULT

        # sanity check : tests if the ckpt is fully consumed at this stage.
        # if warm_start and os.path.isfile(Path(self.checkpoint_dir, 'checkpoint')):
        #     ckpt_load_status.assert_consumed()

        print(f"\n{'':-^{self._progress_bar.terminal_size}}") if verbose not in [
            VerbosityLevel.KEEP_SILENT
        ] else None

        # Train end evaluation epochs
        if self._epoch.value() >= max_epochs:
            print(
                f"Exiting ...\n"
                f"The maximum epochs have already been reached.\n"
                f"Restored epoch == {int(self._epoch.value())}\n"
                f"Allowed epochs == {max_epochs}\n"
            ) if verbose not in [VerbosityLevel.KEEP_SILENT] else None
            callbacks.on_train_end()
        else:
            start_epoch = int(self._epoch.assign_add(1).value())
            logs_init_eval.update({"max_epochs": max_epochs})
            callbacks.on_train_begin(logs=logs_init_eval)
            for epoch in range(start_epoch, max_epochs + 1):
                self._epoch.assign(epoch)
                callbacks.on_epoch_begin(
                    epoch, logs={"samples/train": self._train_samples_count}
                )

                # training
                logs_train = protocol.train(
                    self._data_streams["train"], callbacks, run_eagerly=run_eagerly
                )
                self._train_samples_count = (
                    logs_train.get("samples_done/train", np.nan)
                    if np.isnan(self._train_samples_count)
                    else self._train_samples_count
                )

                # evaluation
                if do_validation:
                    callbacks.on_test_begin(
                        logs={"samples/val": self._eval_samples_count}
                    )
                    logs_eval = protocol.evaluate(
                        self._data_streams["evaluate"],
                        callbacks,
                        run_eagerly=run_eagerly,
                    )
                    self._eval_samples_count = (
                        logs_eval.get("samples_done/val", np.nan)
                        if np.isnan(self._eval_samples_count)
                        else self._eval_samples_count
                    )
                    callbacks.on_test_end()
                else:
                    # when validation data is absent, train metrics are used as val metrics.
                    logs_eval = {
                        k.replace("/train", "/val"): v for k, v in logs_train.items()
                    }
                callbacks.on_epoch_end(
                    epoch,
                    logs={
                        "samples/train": self._train_samples_count,
                        "samples/val": self._eval_samples_count,
                        **logs_train,
                        **logs_eval,
                    },
                )
            callbacks.on_train_end()

            print(f"\n{'':-^{self._progress_bar.terminal_size}}\n") if verbose not in [
                VerbosityLevel.KEEP_SILENT
            ] else None

    def to_json(self):
        config = {
            "logs_dir": self._logs_dir,
            "name": self.name,
            "run_id": self.run_id,
            "data_streams": {
                key: value.to_json() if value is not None else None
                for key, value in self._data_streams.items()
            },
            "models": {
                key: value.to_json() if value is not None else None
                for key, value in self.models.items()
            },
            "checkpointer": self._checkpointer_config,
        }
        return config

    def write_config_to_disk(self):
        config_file = Path(os.path.join(self.run_dir, "params.json"))
        with open(config_file, "w") as cf:
            json.dump(self.to_json(), cf, indent=4, sort_keys=False)

    def _profile_inference_graph(self, data_stream: DataStream, silent=True):
        elements_spec = data_stream.as_dataset().element_spec
        if not isinstance(elements_spec, Iterable):
            elements_spec = make_spec_concrete(elements_spec)
        elif isinstance(elements_spec, dict):
            elements_spec = dict(map(make_spec_concrete, elements_spec.items()))
        elif isinstance(elements_spec, list):
            elements_spec = dict(map(make_spec_concrete, elements_spec))
        else:
            raise ValueError("The element specs received are not in supported format")
        predict_function = tf.function(
            self._protocol.predict_step, input_signature=[elements_spec]
        )

        graph = predict_function.get_concrete_function().graph
        profile_op = ProfileOptionBuilder(ProfileOptionBuilder.float_operation())
        profile_opts = (
            profile_op.with_empty_output().build() if silent else profile_op.build()
        )
        flops_info = profile(graph, options=profile_opts)
        total_parameters = int(np.sum([np.prod(var.shape) for var in graph.variables]))
        flops = flops_info.total_float_ops
        trainable_parameters = int(
            np.sum([np.prod(var.shape) for var in graph.trainable_variables])
        )
        non_trainable_parameters = total_parameters - trainable_parameters
        self.predict_complexity = ComputeComplexity(
            flops=flops,
            trainable_parameters=trainable_parameters,
            non_trainable_parameters=non_trainable_parameters,
        )
        if not silent:
            print(f"\n{'':-^{self._progress_bar.terminal_size}}")
            print(f"\nCompute Complexity [Prediction]:\n")
            print(self.predict_complexity)
            print(f"{'':-^{self._progress_bar.terminal_size}}\n")

    def __str__(self):
        name = "unnamed" if self._name is None else self._name
        return name


if __name__ == "__main__":
    trainer = Trainer(DataStream("cifar10"), name="cifar10Classifier")
    print(str(trainer))