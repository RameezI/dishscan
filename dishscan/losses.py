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

import tensorflow as tf
import numpy as np
from typing import Union, Callable
from .utils import TensorLike
from .utils import capture_params

# Aliases
Loss = tf.keras.losses.Loss
Reduction = tf.keras.losses.Reduction


def reduce_loss(losses, reduction=Reduction.SUM_OVER_BATCH_SIZE):
    """Reduces the individual weighted loss measurements."""
    if reduction == Reduction.NONE:
        loss = losses
    else:
        # loss = tf.reduce_sum(losses)
        loss = tf.reduce_mean(losses) if reduction == Reduction.SUM_OVER_BATCH_SIZE \
            else tf.reduce_sum(losses)
    return loss


def binary_cross_entropy(y_true: TensorLike, y_pred: TensorLike, weights: Union[None, TensorLike] = None,
                         focus_credit: float = 0., gamma_neg: float = 0., gamma_pos: float = 0.,
                         label_smoothing: float = 0.,
                         epsilon=1e-8
                         ):
    """Computes the binary cross entropy loss.
    Standalone usage:
    >>> targets = [[0, 1], [0, 0]]
    >>> scores = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = binary_cross_entropy(targets, scores)
    >>> loss.numpy()
    array([0.916 , 0.714], dtype=float32)
    
    Args:

     y_true:               Ground truth values. shape = `[batch_size, d0, .. dN]`.
     
     y_pred:               The predicted values. shape = `[batch_size, d0, .. dN]`.
     
     weights:              The relative weight for each logit/prediction; the shape must allow
                            broadcasting to the y_true and y_pred
     
     focus_credit:         Provides asymmetric clipping. Adds some constant slack to the p- .This is done to ensure an
                           increased focus, by a constant value, for positive class detection.
    
     gamma_neg:            Controls the weighting of the penalty for (true/false) positive detection.
     gamma_pos:           Controls the weighing of the penalty for (true/false) negative detection.
     
     label_smoothing:      Float in [0, 1]. If > `0` then smooth the labels by
                           squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
                           for the target class and `0.5 * label_smoothing` for the non-target class.
     
     epsilon:               The epsilon used in the calculations.
     
    Returns:
      Binary cross entropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    
    # labels = tf.clip_by_value(labels, epsilon, 1. - epsilon)
    
    if weights is not None:
        raise NotImplementedError
    
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    if label_smoothing > 0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    
    labels, predictions = y_true, y_pred
    anti_labels, anti_predictions = (1 - labels), (1 - predictions)
    
    if focus_credit > 0:
        anti_predictions += focus_credit
        anti_predictions = tf.clip_by_value(anti_predictions, 0, 1)
    
    # Basic CE calculation
    bce = labels * tf.math.log(predictions + epsilon)
    bce += anti_labels * tf.math.log(anti_predictions + epsilon)
    
    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        xs_pos = tf.stop_gradient(predictions * labels)
        xs_neg = tf.stop_gradient(anti_predictions * anti_labels)
        asymmetric_w = tf.stop_gradient(tf.pow(1 - xs_pos - xs_neg,
                                               gamma_pos * labels + gamma_neg * anti_labels))
        bce *= asymmetric_w
    # bce = tf.reduce_mean(bce, axis=-1)  # mean over all classes
    return -bce


def cross_entropy(y_true: TensorLike, y_pred: TensorLike, weights: Union[None, TensorLike] = None,
                  label_smoothing: float = 0.0, axis=-1):
    """Computes the binary cross entropy loss.
        Standalone usage:
    >>> labels = [[0, 1], [0, 0]]
    >>> predictions = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = cross_entropy(labels, predictions)
    >>> loss.numpy()
    
    array([0.916 , 0.714], dtype=float32)
    Args:
      y_true:               Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred:               The predicted values. shape = `[batch_size, d0, .. dN]`.
      weights:              The relative weight for each logit/prediction; the shape must allow
                            broadcasting to the y_true and y_pred
      label_smoothing:      Float in [0, 1]. If > `0` then smooth the labels by
                            squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
                            for the target class and `0.5 * label_smoothing` for the non-target class.
      axis:                 The axis over which the cross entropy is calculated; default value is -1
                            This is normally the class/category axis.
                            
    Returns:
      Binary cross entropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    
    if weights is not None:
        raise NotImplementedError
    
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    if label_smoothing > 0.0:
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    
    epsilon_ = tf.constant(np.finfo(float).eps, y_pred.dtype)
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
    output = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
    xentropy = -tf.reduce_sum(y_true * tf.math.log(output), axis)
    return xentropy


def hamming_distance(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    """Computes hamming distance.
    Hamming distance is for comparing two binary strings.
    It is the number of bit positions in which two bits
    are different.
    Args:
        y_true: target values.
        y_pred: predicted values.
    Returns:
        hamming distance: float.
    Usage:
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.int32)
    >>> y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=np.int32)
    >>> hamming_distance(y_true, y_pred).numpy()
    0.3
    """
    result = tf.not_equal(y_true, y_pred)
    not_eq = tf.reduce_sum(tf.cast(result, tf.float32))
    ham_distance = tf.math.divide_no_nan(not_eq, len(result))
    return ham_distance


class CrossEntropy(Loss):
    def __init__(self, transform: Callable = lambda x: x, label_smoothing=0.0,
                 reduction: Reduction = Reduction.AUTO,
                 name: str = 'cross_entropy'):
        """Initializes `CrossEntropy` instance.
          Args:

            transform:          The transformation applied to the scores' tensor before calculation the loss

            label_smoothing:    Float in [0, 1]. When 0, no smoothing occurs. When > 0,
                                we compute the loss between the predicted labels and a smoothed version
                                of the true labels, where the smoothing squeezes the labels towards 0.5.
                                Larger values of `label_smoothing` correspond to heavier smoothing.

            reduction:          Type of `tf.keras.losses.Reduction` to apply to
                                loss. Default value is `AUTO`. `AUTO` indicates that the reduction
                                option will be determined by the usage context. For almost all cases
                                this defaults to `SUM_OVER_BATCH_SIZE`.

            name:               Name for the op. Defaults to 'binary_cross_entropy'.
          """
        super().__init__(reduction, name)
        self._transform = transform
        self._label_smoothing = label_smoothing
    
    def __call__(self, targets: TensorLike, scores: TensorLike, weights: TensorLike = None):
        predictions = self._transform(scores)
        losses = cross_entropy(targets, y_pred=predictions, weights=weights,
                               label_smoothing=self._label_smoothing,
                               axis=-1)
        reduced_loss = reduce_loss(losses)
        return reduced_loss


class BinaryCrossEntropy(Loss):
    def __init__(self, transform: Callable = lambda x: x,
                 focus_credit: float = 0.0, gamma_neg: float = 0.0, gamma_pos: float = 0.0,
                 label_smoothing: float = 0.0,
                 reduction=Reduction.AUTO, name='binary_cross_entropy'):
        """Initializes `BinaryCrossEntropy` instance.
          
          Args:
            transform:          The transformation applied to the scores' tensor before calculation the loss.
     
            focus_credit:       Provides asymmetric clipping. Adds some constant slack to the p- .This is done
                                to ensure an increased focus, by a constant value, for positive class detection.
    
            gamma_pos:          Controls the weighing of the penalty for (true/false) negative detection.
            
            gamma_neg:          Controls the weighting of the penalty for (true/false) positive detection.
      

            label_smoothing:    Float in [0, 1]. When 0, no smoothing occurs. When > 0,
                                we compute the loss between the predicted labels and a smoothed version
                                of the true labels, where the smoothing squeezes the labels towards 0.5.
                                Larger values of `label_smoothing` correspond to heavier smoothing.

            reduction:          Type of `tf.keras.losses.Reduction` to apply to
                                loss. Default value is `AUTO`. `AUTO` indicates that the reduction
                                option will be determined by the usage context. For almost all cases
                                this defaults to `SUM_OVER_BATCH_SIZE`.

            name:               Name for the op. Defaults to 'binary_cross_entropy'.
          """
        super().__init__(reduction, name)
        self._transform = transform
        self._focus_credit = focus_credit
        self._gamma_pos = gamma_pos
        self._gamma_neg = gamma_neg
        self._label_smoothing = label_smoothing
    
    def __call__(self, targets: TensorLike, scores: TensorLike, weights: TensorLike = None):
        predictions = self._transform(scores)
        losses = binary_cross_entropy(targets, y_pred=predictions, weights=weights,
                                      focus_credit=self._focus_credit,
                                      gamma_pos=self._gamma_pos,
                                      gamma_neg=self._gamma_neg,
                                      label_smoothing=self._label_smoothing)
        reduced_loss = reduce_loss(losses)
        return reduced_loss


class HammingLoss(Loss):
    def call(self, y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
        """Computes hamming loss.
        Hamming loss is the fraction of wrong labels to the total number
        of labels.In a multi-label classification, hamming loss penalizes
        only the individual labels.
        Args:
            y_true: actual target value.
            y_pred: predicted target value.
        Returns:
            hamming loss: float.
        """
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


class CumulativeMAE(tf.keras.losses.Loss):
    
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self, accumulation_axis=-1, **kwargs):
        super().__init__(name=kwargs.pop('name', None))
        self.axis = accumulation_axis
        self.tf2_mae_loss = tf.keras.losses.MeanAbsoluteError(**kwargs)
    
    def call(self, y_true: TensorLike, y_pred: TensorLike,
             sample_weight: Union[None, TensorLike] = None) -> tf.Tensor:
        """Computes cumulative mae loss.
        Args:
            y_true: actual target value.
            y_pred: predicted target value.
            sample_weight: Wight for each sample, when None all samples carry an equal weight.
        Returns:
            cumulative mae loss: float.
        """
        y_true = tf.reduce_sum(y_true, axis=self.axis)
        y_pred = tf.reduce_sum(y_pred, axis=self.axis)
        return self.tf2_mae_loss(y_true, y_pred, sample_weight)