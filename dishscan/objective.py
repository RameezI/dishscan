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
from typing import Sequence
from .metrics import Metric, Mean

# Aliases
Reduction = tf.keras.losses.Reduction


def scaled_loss(loss_value):
    """Scales and returns the given loss value by the number of replicas.
    Sum_over_batch size reduction over replicas:
        losses = [sum(loss_k)/batch_size_k for loss_k in replicas]
     :: actual_batch_size = k*batch_size
     => correction:  loss = loss/k
    """
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        loss_value *= (1. / num_replicas)
    return loss_value


class DistributedLossWrapper:
    """A container that wraps the loss object to correctly
     perform reduction while training on multiple replicas in sync"""
    
    def __init__(self, loss: tf.keras.losses.Loss):
        super(DistributedLossWrapper, self).__init__()
        loss._allow_sum_over_batch_size = True
        self._loss = loss
    
    def __call__(self, *args, **kwargs):
        if self._loss.reduction in [Reduction.SUM_OVER_BATCH_SIZE]:
            loss = scaled_loss(self._loss(*args, **kwargs))
        else:
            loss = self._loss(*args, **kwargs)
        return loss


class Objective(tf.Module):
    """This class defines a container for the training objective and associated metrics to measure its progress.The
     objective  class also describes the optimizer used to optimize the loss function. This class also applies
     compensations for distribution of the batch and the loss function across multiple devices. For example, when
     reducing the losses across multiple devices using sum_over_batch_size, it ensures that the correct global
     batch_size is used and not the device-local batch_size. The motivation behind this abstraction is to enable
     users write loss/optimizer/metrics for a single device,without worrying about the distribution strategies,
     while this class applies the necessary adjustments.
    
    """
    
    def __init__(self, loss: tf.keras.losses.Loss, optimizer: tf.keras.optimizers.Optimizer, *,
                 name: str = 'objective',
                 metrics: Sequence[Metric] = tuple()):
        super().__init__(name)
        self._optimizer = optimizer
        self._loss: DistributedLossWrapper = DistributedLossWrapper(loss)
        self._optimizer: tf.keras.optimizers.Optimizer = optimizer
        self._loss_metric = Mean(name='loss')
        self._metrics = list(metrics) + [self._loss_metric]
    
    @property
    def name(self):
        return self._name
    
    @property
    def metrics(self):
        return self._metrics
    
    @property
    def optimizer(self):
        """
        Returns:
        The optimizer for the composed loss object of the objective
        """
        return self._optimizer
    
    def compute_loss(self, *args, **kwargs) -> tf.Tensor:
        """
        Computes the overall loss and updates the average, across all batches, loss metric.
        The class is distribution friendly i.e. it takes into account the corrections required
        for computing and aggregating losses over multiple replicas.
    
        Args:
            *args:                      The arguments accepted by the loss object
            **kwargs:                   Additional keyword args thar must be passed on to the loss object
    
        Returns:
        
        """
        loss_value = self._loss(*args, **kwargs)
        self._loss_metric.update(loss_value)
        return loss_value
        
    def update_metrics(self, *args, exclude: Sequence[str] = ('loss',), **kwargs) -> None:
        metrics = filter(lambda x: True if x.name not in exclude else False, self.metrics)
        for metric in metrics:
            metric.update(*args, **kwargs)
        # map(lambda x: x.update(*args, **kwargs), metrics)