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

import tensorflow as tf
import functools
from typing import Tuple, Union, Callable, Dict
from .protocol import Protocol
from .objective import Objective
from .losses import BinaryCrossEntropy
from .lr_schedulers import LinearWarmupCosineDecay

from .metrics import AccuracyScore, sigmoid_transform,\
    multihot_transform, AveragingMode, AveragePrecisionScore


def compose(*functions: Callable) -> Callable:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


class MLCProtocol(Protocol):
    """Training protocol for MultiLabelClassification (MLC)
    A protocol specifies the computations of the train, evaluate
    and predict steps for the trainer.
    """

    INITIAL_LR: float = 1E-3
    ITERATIONS_ESTIMATE: int = 23650
    WARMUP_STEPS: int = 200

    def __init__(self, supervision_keys: Tuple[str, str]):
        """
        Args:
            supervision_keys:   An ordered pair of strings; where the first element represents
                                the key for the input data while the second elements is the key
                                for the label.
        """
        super().__init__()
        self.supervision_keys = supervision_keys

    
    @staticmethod
    def configure():
        base_lr = MLCProtocol.INITIAL_LR
        warmup_steps = MLCProtocol.WARMUP_STEPS
        decay_steps = MLCProtocol.ITERATIONS_ESTIMATE - warmup_steps
        scheduler = LinearWarmupCosineDecay(base_lr, warmup_steps, decay_steps, alpha=1e-3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
        
        # loss
        binary_xentropy = BinaryCrossEntropy(transform=tf.nn.sigmoid, focus_credit=0.05, gamma_neg=4)
        
        # metrics
        transform = compose(sigmoid_transform, multihot_transform)
        subset_accuracy = AccuracyScore(transform, subset_mode=True,
                                        averaging_mode=AveragingMode.SAMPLE,
                                        name='subset_acc')
        average_precision = AveragePrecisionScore(sigmoid_transform,
                                                  averaging_mode=AveragingMode.MICRO,
                                                  name='mAP')
        
        objective = Objective(binary_xentropy, optimizer=optimizer,
                              metrics=[subset_accuracy, average_precision])
        return objective

    def train_step(self, batch: Dict[str, tf.Tensor]) -> None:
        """
        Args:
            batch:                      A batch of data with various features
        """

        data, labels = batch[self.supervision_keys[0]], batch[self.supervision_keys[1]]

        # aliases
        encoder = self.models["encoder"]
        decoder = self.models["decoder"]
        objective = self.objective
        optimizer = self.objective.optimizer

        with tf.GradientTape() as tape:
            encoding = encoder(data, training=True)
            predictions = decoder(encoding, training=True)
            regularization_loss = encoder.losses + decoder.losses
            loss = objective.compute_loss(labels, predictions) + regularization_loss

        # apply optimization to the trainable variables
        parameters = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, parameters)
        optimizer.apply_gradients(zip(gradients, parameters))
        objective.update_metrics(labels, predictions)

    def evaluate_step(self, batch: Dict[str, tf.Tensor]):

        data, labels = batch[self.supervision_keys[0]], batch[self.supervision_keys[1]]

        # aliases
        encoder = self.models["encoder"]
        classifier = self.models["decoder"]
        objective = self.objective

        encoding = encoder(data, training=False)
        predictions = classifier(encoding, training=False)
        regularization_loss = encoder.losses + classifier.losses

        # calculate loss and update loss metrics
        objective.compute_loss(labels, predictions) + regularization_loss
        objective.update_metrics(labels, predictions)

    # The default serving signature
    def predict_step(self, batch: Union[tf.Tensor, Dict[str, tf.Tensor]]):
        data = batch[self.supervision_keys[0]] if isinstance(batch, Dict) else batch
        encoding = self.models["encoder"](data, training=False)
        predictions = self.models["decoder"](encoding, training=False)
        return predictions