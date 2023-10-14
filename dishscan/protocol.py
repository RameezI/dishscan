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

import inspect
import logging
from pathlib import Path
from typing import Collection
import tensorflow as tf
from tensorflow import distribute
from typing import Any, Dict, List, Union
from .objective import Objective
from .datastreams import DataStream
from .utils import snake_to_camel

# Aliases
CallbackList = tf.keras.callbacks.CallbackList


class Protocol(tf.Module):
    ClusterCoordinator = distribute.experimental.coordinator.ClusterCoordinator

    def __init__(self):
        name = snake_to_camel(self.__module__, splitter=".")
        super(Protocol, self).__init__(name=name)
        self._cluster_coordinator: Union[None, Protocol.ClusterCoordinator] = None
        self._models: Union[None, Dict[str, tf.keras.Mode]] = None
        self._objectives: Union[None, Dict[str, Objective]] = None
        self._metrics: List[tf.keras.metrics.Metric] = []
        self._distributor = None

        # boolean flags
        self._base_strategy_initialized: bool = True
        self._run_eagerly: bool = False
        self._is_compiled: bool = False

        # cached step_tf functions
        self.train_step_tf = None
        self.evaluate_step_tf = None
        self.predict_step_tf = None

    @property
    def is_initialized(self):
        try:
            self._base_strategy_initialized
        except AttributeError:
            raise RuntimeError(
                "It looks like you are subclassing `Strategy` and forgot to call"
                "`super(YourClass, self).__init__()`.\n"
                "Always start your subclassed strategy with this line."
            )
        return self._base_strategy_initialized

    def compiled(self):
        return self._is_compiled

    @property
    def cluster_coordinator(self):
        return self._cluster_coordinator

    @property
    def metrics(self):
        self._metrics = [
            metric
            for objective in self.objectives.values()
            for metric in objective.metrics
        ]
        return self._metrics

    @property
    def objective(self):
        objective: Union[None, Objective] = None
        if isinstance(self._objectives, dict) and len(self._objectives) > 1:
            raise AttributeError(
                f"Invalid use of attribute objective. This property is only populated in "
                f"case of a single objective, while multiple objectives were spotted.\n "
                f" Please make use of `objectives` property instead \n"
                f" Spotted objectives: {list(self._objectives.keys())}"
            )
        elif isinstance(self.objectives, dict) and len(self._objectives) == 1:
            objective = [value for value in self._objectives.values()][0]
        return objective

    @property
    def objectives(self):
        return self._objectives

    @property
    def model(self):
        model: Union[None, tf.keras.Model] = None
        if isinstance(self._models, dict) and len(self._models) > 1:
            raise AttributeError(
                f"Invalid use of attribute model. This property is only populated in case of a"
                f"single model setting. While, multiple models setting is spotted.\n"
                f" Please make use of `models` instead \n"
                f"Spotted models keys:{list(self._models.keys())}"
            )

        elif isinstance(self._models, dict) and len(self._models) == 1:
            model = [value for value in self._models.values()][0]
        return model

    @property
    def models(self):
        return self._models

    def reset_metrics(self):
        if self._metrics:
            [metric.reset() for metric in self._metrics]

    def _set_models(self, value: Union[Dict[str, tf.keras.Model], tf.keras.Model]):
        models = value if isinstance(value, dict) else {"model": value}
        self._assert_type_conformity(models, expected_type=tf.keras.Model)
        self._models = models

    def _set_objectives(self, value: Union[Dict[str, Objective], Objective]):
        objectives = value if isinstance(value, dict) else {"objective": value}
        self._assert_type_conformity(objectives, expected_type=Objective)
        self._objectives = objectives

    @staticmethod
    def _assert_type_conformity(collection: Dict[str, Any], expected_type: type):
        faulty_keys = filter(
            lambda value: not isinstance(value, expected_type), collection.values()
        )
        if list(faulty_keys):
            faulty_types = [
                list(map(lambda x: x.__name__, map(type, v)))
                for k, v in collection.items()
                if k in faulty_keys
            ]
            faulty_formulations = [
                f"{k}:{v}" for k, v in zip(faulty_keys, faulty_types)
            ]
            raise ValueError(
                f"Cannot set objectives!\n"
                f"One or more objective does match the expected type restrictions: \n"
                f"Expected: An objective of type: `{expected_type}`\n"
                f"Received: {faulty_formulations}"
            )

    def _compile_compliance_check(self, models):
        params = inspect.signature(self.configure).parameters.values()
        unsupported_kinds = [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ]

        if any([param.kind in unsupported_kinds for param in params]):
            _unsupported_kinds_str = [str(kind) for kind in unsupported_kinds]
            raise TypeError(
                f"Unsupported arguments kind spotted in the `configure_objective`"
                f" method of the strategy class:{str(type(self).__name__)}.\n"
                f"The following kinds are not supported:\n"
                f"{_unsupported_kinds_str}"
            )

        params = list(filter(lambda x: x.kind not in unsupported_kinds, params))
        required_named_args = [
            param.name for param in params if param.default == inspect.Parameter.empty
        ]

        strategy_module: List[str] = str(self.__module__)
        if (
            isinstance(models, dict)
            and all([isinstance(model, tf.keras.Model) for model in models.values()])
            and all([isinstance(key, str) for key in models.keys()])
        ):

            if not all([arg in models.keys() for arg in required_named_args]):
                raise RuntimeError(
                    f"Unable to populate models!\n"
                    f"The model dictionary handed over to the compile function does not comply with"
                    f" the `configure` signature of the `{strategy_module}` module.\n"
                    f"The method must accept (as arguments) the same names as the keys in the model"
                    f" dictionary of the trainer instance.\n"
                    f"A subset of these alias also constitutes a valid argument list!\n"
                    f"The discrepancy:\n"
                    f"`configure` signature: {inspect.signature(self.configure)}\n"
                    f"model keys: {list(models.keys())}"
                )
        else:
            raise ValueError(
                "The first argument of the compile must be a dictionary of tf.keras.Model(s)"
            )
        return required_named_args

    def get_global_batch_size(self, batch, batch_idx=0):
        distributor = self._distributor
        size = set([tf.shape(k)[batch_idx].numpy() for k in batch.values()])
        if not len(size) == 1:
            raise RuntimeError(
                f"Cannot estimate batch size!"
                f" Spotted different values at the batch index for different features,"
                f" all features must have the same dimension at the batch index.\n"
                f" Found {len(size)} unique batch lengths {size} at index={batch_idx}"
            )
        else:
            # TODO: Apply correction based on the distributor!
            size = next(iter(size))
        return size

    def _enable_auto_reduction(self):
        if isinstance(self._, dict):
            for key in self._mo:
                self._objectives[key]._losses._allow_sum_over_batch_size = True
        else:
            self.model.loss._allow_sum_over_batch_size = True

    def compile(
        self,
        models: Dict[str, tf.keras.Model],
    ):
        """
        The protocol compilation perform following tasks:
            1- Create loss functions, metrics and optimizers
               grouped together in organization units `objectives`.
            2- Instantiate the objective that the protocol will be optimizing.
            3- Links the instantiated models from the trainer with the protocol.
            4- Compiles the train/validate/predict steps into a tensorflow graph.
            5- Caches the the distribution strategy from the outer-scope

        Args:
            models:  A dictionary of instantiated models that are linked to the protocol.
        """
        self._distributor = tf.distribute.get_strategy()
        required_args = self._compile_compliance_check(models)
        required_models = {
            key: model for key, model in models.items() if key in required_args
        }

        self._set_models(models)  # Link models from the outer scope with the protocol.
        self._set_objectives(
            self.configure(**required_models)
        )  # set objective/objectives.

        # compiled_step functions for train and evaluate
        self.train_step_tf = tf.function(self.train_step)
        self.evaluate_step_tf = tf.function(self.evaluate_step)
        # self.predict_step_tf = tf.function(self.predict_step)
        self._is_compiled = True

    @staticmethod
    def configure(**kwargs: tf.keras.Model) -> Union[Dict[str, Objective], Objective]:
        """
        Constructs a learning objective for the trainer. The method optionally accepts models, using their aliases as
         keyword/named arguments. Any model that need modification/inspection can be listed as named argument in the
        `configure` method, using its alias registered with the trainer.
        Args:
            **kwargs:     Accepts (zero or more) models provided as keyword arguments using their aliases.
        Returns:
            An Objective or a named collection of such objectives.
        """
        raise NotImplementedError(
            """Lacking a configure method!
             The training strategy does not provide a `configure` method. A strategy without this method cannot
             be actualized as the objective setting cannot be realized. Please provide a `configure` method
             for your strategy, which must return an objective."""
        )

    def train_step(self, input):
        raise NotImplementedError(
            """Lacking `train_step` definition!
            The training strategy does not provide a `train_step` method. A strategy without this method is unable
            be perform training/learning. Please provide a `train_step` method for your strategy, which describes
            a single step for training the underlying models."""
        )

    def evaluate_step(self, input):
        raise NotImplementedError(
            """Lacking `evaluate_step` definition!
            The strategy does not provide an `evaluate_step` method. A strategy without this method is unable to
            determine the evaluation logic. Please provide an `evaluate_step` for your strategy, which describes
            a single step for evaluating the underlying model/s on a single batch."""
        )

    def predict_step(self, input):
        raise NotImplementedError(
            """Lacking `predict_step` definition!
            The strategy does not provide a `predict_step` method. A strategy without this method is unable
            to determine the prediction logic. Please provide a `predict_step` method for your strategy,
            which describes a single step for predicting based on the the underlying trained model/s."""
        )

    def train(
        self,
        dataset: DataStream,
        callbacks: CallbackList = CallbackList(),
        run_eagerly=False,
        postfix="train",
        **kwargs,
    ) -> Dict[str, any]:

        examples_processed = 0
        self.reset_metrics()

        logs = {}
        for step, batch in enumerate(dataset):
            callbacks.on_train_batch_begin(step)
            if run_eagerly:
                self._distributor.run(self.train_step, args=[batch], kwargs=kwargs)
            else:
                self._distributor.run(self.train_step_tf, args=[batch], kwargs=kwargs)

            examples_processed += self.get_global_batch_size(batch)

            logs["samples_done"] = examples_processed
            logs.update(
                {f"{metric.name}/{postfix}": metric.result() for metric in self.metrics}
            )
            callbacks.on_train_batch_end(step, logs)
        return logs

    def evaluate(
        self,
        dataset: DataStream,
        callbacks: CallbackList = CallbackList(),
        run_eagerly=False,
        postfix="val",
        **kwargs,
    ) -> Dict[str, any]:

        examples_processed = 0
        self.reset_metrics()

        logs = {}
        for step, batch in enumerate(dataset):
            callbacks.on_test_batch_begin(step)
            if run_eagerly:
                self._distributor.run(self.evaluate_step, args=[batch], kwargs=kwargs)
            else:
                self._distributor.run(
                    self.evaluate_step_tf, args=[batch], kwargs=kwargs
                )

            examples_processed += self.get_global_batch_size(batch)

            logs["samples_done"] = examples_processed
            logs.update(
                {f"{metric.name}/{postfix}": metric.result() for metric in self.metrics}
            )
            callbacks.on_test_batch_end(step, logs)
        return logs

    def export(
        self, destination: Path, signature: Collection[tf.TensorSpec] = ()
    ) -> None:
        """Export this protocol as as a saved model"""

        if self._is_compiled and signature:
            self.predict_step_tf = tf.function(self.predict_step, input_signature=signature)
            tf.saved_model.save(self, destination, signatures=self.predict_step_tf)

        elif self._is_compiled:
            logging.warning(
                "\n\nA serving signature is not prescribed and hence will not be exported!"
                "\ncontinuing export without a `default_serving_signature`, the model cannot be used for inference ... "
            )
            tf.saved_model.save(self, destination)

        else:
            logging.warning(
                "\n\nCannot export a protocol that is not yet compiled!"
                "continuing without export ... "
            )
            raise NotImplemented

    def __str__(self):
        return self.name