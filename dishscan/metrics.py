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

import functools
from functools import partial
from abc import ABC, abstractmethod
from enum import Enum
from typing import \
    Callable, Union, Tuple, Dict, Sequence, Any

import numpy as np
import tensorflow as tf
from dishscan.utils import TensorLike


def compose(*functions: Callable) -> Callable:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)

# Aliases
tnp = tf.experimental.numpy


class Metric(tf.Module, ABC):
    """
    Base class for all Metrics.
    
    Examples:
                To be Populated.
    """
    
    def __init__(self, name: str, dtype: tf.DType = tf.float32):
        super().__init__(name)
        self.dtype = dtype
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to it's initial state.
        Usually, this is called at the start of each epoch.
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Updates the metrics state using the passed batch output.
        Usually, this is called once for each batch.
        Args:
            output: the output from the protocol process function.
        """
        pass
    
    @abstractmethod
    def result(self) -> Any:
        """
        Provides the accumulated/collected result for the metric.
        Usually, this is called at the end of each epoch.
        Returns:
            Any: the actual quantity of interest. However, if a :class:`~collections.abc.Mapping` is returned,
                 it will be (shallow) flattened form of the metric.
        """
        pass


class Mean(Metric):
    def __init__(self, *, axis=None, keep_dims: bool = False, name: str = 'mean', dtype=tf.float32):
        super().__init__(name, dtype=dtype)
        self.axis = axis
        self._value_sum = None
        self._count = None
        self.keep_dims = keep_dims
    
    @staticmethod
    def _create_variables(shape: Tuple):
        value_sum = tf.Variable(tf.zeros(shape), trainable=False, shape=shape)
        count = tf.Variable(0.0, trainable=False, shape=tuple())
        return value_sum, count
    
    def reset(self) -> None:
        self._value_sum.assign(tf.zeros(self._value_sum.shape, self.dtype)) \
            if self._value_sum is not None else None
        
        self._count.assign(tf.zeros(self._count.shape, self.dtype)) \
            if self._value_sum is not None else None
    
    def _variables_shape(self, input_shape):
        if self.keep_dims:
            shape = input_shape
            shape = tf.tensor_scatter_nd_update(shape, [[self.axis]], [1])
        elif self.axis is None:
            shape = tuple()
        else:
            axis = self.axis if self.axis >= 0 \
                else (len(input_shape) + self.axis)
            axes = [_axis for _axis in range(len(input_shape)) if _axis != axis]
            shape = tf.gather(input_shape, axes)
        return shape
    
    def update(self, values: TensorLike) -> None:
        value_sum = tf.reduce_sum(values, axis=self.axis, keepdims=self.keep_dims)
        count = tf.cast(tf.size(values) if self.axis is None
                        else tf.shape(values)[self.axis], dtype=self.dtype)
        
        if self._value_sum is None or self._count is None:
            shape = self._variables_shape(tf.shape(values))
            self._value_sum, self._count = self._create_variables(shape)
        
        self._value_sum.assign_add(value_sum)
        self._count.assign_add(count)
    
    def result(self) -> TensorLike:
        mean = tf.constant(np.nan)
        mean = self._value_sum / self._count \
            if self._count is not None else mean
        return mean


class AveragingMode(Enum):
    SAMPLE = 'sample'
    MICRO = 'micro'
    MACRO = 'macro'
    WEIGHTED = 'weighted'


def onehot_transform(x: TensorLike, axis=-1) -> tf.Tensor:
    indices = tf.argmax(x, axis=axis)
    onehot_predictions = tf.one_hot(indices, depth=x.shape[axis])
    return onehot_predictions


def sigmoid_transform(x: TensorLike) -> tf.Tensor:
    sigmoid_predictions = tf.sigmoid(x)
    return sigmoid_predictions


def multihot_transform(x: TensorLike, threshold=0.5) -> tf.Tensor:
    multihot_encoding = tf.cast(x > threshold, x.dtype)
    return multihot_encoding


def compute_confusion_matrix(y_true: TensorLike, y_pred: TensorLike,
                             bifurcators: int = 1,
                             weights=tf.constant(1.0),
                             ) -> Tuple[Dict[str, tf.Tensor], Sequence]:
    """Update the confusion matrix variables.
    
    To compute TP/FP/TN/FN, we  measure the classifier at given thresholds `t`
      C(t) = (predictions >= t) at each threshold 't_k'. So we have
      TP(t_k) = sum( C(t_k) * true_labels )
      FP(t_k) = sum( C(t_k) * false_labels )
      
    This function is adapted from keras  confusion matrix update function, as it provides an efficient
    algorithm for binning the variables at various thresholds. The original function can  be found under:
    https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/utils/metrics_utils.py#L262
    

      Args:
      y_true:                   A floating point `Tensor` whose shape matches `y_pred`. Will be cast
                                to `bool`.
      y_pred:                   A floating point `Tensor` of arbitrary shape and whose values are in
                                the range `[0, 1]`.
      bifurcators:              The number of dividers that bifurcates the one dimensional space, [0 ,1] into
                                classification subspaces. Based on the `bifurcators` value specific thresholds
                                are applied for calculating the confusion variables. The thresholds are computed
                                to be equidistant from each other and from the end-points.
                                
                                The bifurcators value is strictly a positive integer, by default is set to 1,
                                this divides the line at 0.5
                                                        0----------(0.5)----------1
                                                                    \
                                                                     \--> bifurcation
      Examples:
                                y_true = [1, 1, 0, 1]
                                y_pred = [0.0, 0.2, 0.6, 1.0]
                                
                                thresholds_count == 1:
                                    thresholds = np.linspace(0.0, 1.0, num=thresholds_count+2]
                                                 ==> [0, 0.5, 1.0], (0.0, 1.0], (1.0, 2.0]
                                    num_buckets = 2 == len(thresholds)-1
                                    bucket_index(y_pred) = tf.math.floor(y_pred * num_buckets)

    Returns:
      
      confusion variables :     A dict with keys:
                                ('true_positives', 'false_positives','true_negatives','false_negatives')
                                evaluated at specified thresholds.These variables are calculated for each class
                                independently and thus have a shape: (thresholds_count, labels_count)
                                
      thresholds:               The thresholds at which the confusion variables are evaluated. The threshold
                                implies that for a positive detection the scores must bee strictly greater than
                                the specified threshold.
    """
    if bifurcators < 1:
        raise ValueError('Invalid value for bifurcators!\n'
                         'The bifurcators is strictly a positive integer i.e. must be >0'
                         f'Found: {bifurcators}')
    
    thresholds = np.linspace(0.0, 1.0, num=bifurcators + 2, endpoint=True)
    num_buckets = len(thresholds)
    
    y_pred = y_pred - np.finfo(np.float16).eps
    # ensure values stays in the [0, 1] range
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0.0, clip_value_max=1.0)
    
    true_labels = tf.multiply(y_true, weights)
    false_labels = tf.multiply((1.0 - y_true), weights)
    total_true_labels = tf.reduce_sum(true_labels, axis=0)
    total_false_labels = tf.reduce_sum(false_labels, axis=0)
    
    bucket_indices = tf.math.floor(y_pred * (num_buckets - 1))
    bucket_indices = tf.cast(bucket_indices, tf.int32)
    
    true_labels = tf.transpose(true_labels)
    false_labels = tf.transpose(false_labels)
    
    # Fill buckets
    bucket_indices = tf.transpose(bucket_indices)
    
    def gather_buckets(args):
        data, ids = args
        return tf.math.unsorted_segment_sum(data=data, segment_ids=ids,
                                            num_segments=num_buckets)
    
    tp_buckets = tf.vectorized_map(gather_buckets, (true_labels, bucket_indices))
    fp_buckets = tf.vectorized_map(gather_buckets, (false_labels, bucket_indices))
    
    # calculate tp/fp/tn/fn
    tp = tf.transpose(tf.cumsum(tp_buckets, reverse=True, axis=1))
    fp = tf.transpose(tf.cumsum(fp_buckets, reverse=True, axis=1))
    tn = total_false_labels - fp
    fn = total_true_labels - tp
    
    return {'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
            }, thresholds


class ClassificationMetric(Metric, ABC):
    """
    This is an abstract base class for all classification metrics.

      Args:
      name:               The name of the metric. This is a position only argument, which means to
                          instantiate this class a metric name must be assigned first.

      transform:          Transforms the prediction tensor; by default a sigmoid transforms is applied.

      averaging_mode:     The  accumulating mode for the metric


                          'sample':      Computes the metric for each `sample` and report an average overall samples.

                           'micro':      Calculates the metric by regarding each score in the sample as an independent
                                         prediction, contributing equally to the final metric. The average is
                                         performed over all predictions globally.

                           'macro':      Calculates the metric for each class separately, and reports their unweighted
                                         mean. This does not take label-imbalance into account.

                           'weighted':   Calculate metrics for each class, and reports their weighted, by support
                                        (the number of true instances for each label) average. This alters ‘macro’
                                        to account for label imbalance; it can result in an F-score that is not
                                        between precision and recall.

                           None:        Provides metrics for each class independently. By default, this metric
                                         returns the class-wise accuracy scores.

      **kwargs             Any additional arguments accepted by the base class

    """
    
    def __init__(self, transform: Callable = lambda x: x,
                 averaging_mode: Union[AveragingMode, None] = None, *,
                 name: str, dtype: tf.DType = tf.float32
                 ):
        super().__init__(name, dtype)
        self._transform = transform
        self.average_mode = averaging_mode
        self.dtype = dtype
        
        self._value_sum = None  # accumulates the value
        self._count = None  # accumulates the count
    
    @staticmethod
    def _check_shape(y_true: TensorLike, y_pred: TensorLike, weights: TensorLike):
        msg = 'Inconsistent shapes spotted when calculating classification metric!\n'
        shapes = [(y_true, ('N', 'L')),
                  (y_pred, ('N', 'L'))]
        tf.debugging.assert_shapes(shapes, message=msg, summarize=True)

        tf.debugging.assert_rank_in(weights, [0, 1, 2], message=msg)

        pass  # TODO: CHeck if this can be ingested.
        # if len(weights.shape) == 2:
        #     tf.debugging.assert_shapes([(weights, (None, 'L'))])
        #
        # if len(weights.shape) == 1:
        #     tf.debugging.assert_shapes([(weights, 'N')])
        #
    
    def _preprocess(self, y_true: TensorLike, scores: TensorLike,
                    weights: TensorLike) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """
        Converts inputs to tensors and preform shape checking
        """
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        scores = tf.convert_to_tensor(scores, dtype=self.dtype)
        weights = tf.convert_to_tensor(weights, dtype=self.dtype)
        self._check_shape(y_true, scores, weights)
        # Transform the scores to predictions
        y_pred = self._transform(scores)
        if tf.rank(weights) == 1:
            weights = tf.reshape(weights, (-1, 1))
            weights = tf.cast(weights, self.dtype)
        return y_true, y_pred, weights
    
    def _create_variables(self, shape):
        """ These variables are used to track accuracy score for the case of `sample` and
         `micro` averaging.
         """
        value_sum = tf.Variable(tf.zeros(shape), trainable=False, dtype=self.dtype)
        count = tf.Variable(tf.zeros(shape), trainable=False, dtype=self.dtype)
        return value_sum, count
    
    def reset(self) -> None:
        self._value_sum.assign(tf.zeros(tf.TensorShape(tf.shape(self._value_sum)), self.dtype)) \
            if self._value_sum is not None else None
        
        self._count.assign(tf.zeros(tf.TensorShape(tf.shape(self._count)), self.dtype)) \
            if self._value_sum is not None else None
    
    def result(self) -> TensorLike:
        
        if self.average_mode in [AveragingMode.MACRO, AveragingMode.WEIGHTED]:
            if self.average_mode == AveragingMode.WEIGHTED:
                count = tf.reduce_sum(self._count)
                reduce_op = tf.reduce_sum
            else:
                count = self._count
                reduce_op = tf.reduce_mean
            
            # scores = tf.where(count > tf.constant(0.),
            #                   self._value_sum / count, tf.zeros_like(self._value_sum))
            scores = tf.math.divide_no_nan(self._value_sum, count)
            scores = reduce_op(scores, axis=-1)
        else:
            scores = tf.math.divide_no_nan(self._value_sum, self._count)
        
        if tf.size(scores) == 1:
            scores = tf.reshape(scores, ())
        return scores


class PrecisionScore(ClassificationMetric):
    """
      Args:
      transform:          Transforms the prediction tensor; by default a sigmoid transforms is applied.
      
      bifurcators:         The number of equally distributed thresholds between 0 and 1; the default values is 1.
                           This means by default the metric is evaluated at a decision threshold of 0.5.
      
      include_endpoints:   Weather to include the endpoints, i.e. 0 and 1, as thresholds for precision evaluation.
                           By default, the endpoints are not included while reporting the precision scores.

      averaging_mode:     The  accumulating mode for the metric
      
                           'sample':     Calculate metrics for each instance, and report the average subset precision.
                                         (only meaningful for multilabel classification where this differs
                                         from accuracy_score).

                           'micro':      Calculates the precision over all prediction elements by inflating the
                                         prediction and label tensors.

                           'macro':      Calculates mean precision for each class and report their unweighted mean.
                                         This does not take label-imbalance into account.

                           'weighted':   Calculates mean accuracy for each class and reports a weighted, by support
                                         (the number of true instances for each label), average. This alters ‘macro’
                                         to account for label imbalance; it can result in an F-score that is not
                                         between precision and recall.

                           None:        Provides metrics for each class independently. By default, this metric
                                         returns the class-wise accuracy score.

              **kwargs                   Any additional arguments accepted by the base class

      name:               The name of the metric.
    """
    
    def __init__(self, transform: Callable = lambda x: x,
                 bifurcators: int = 1,
                 include_endpoints: bool = False,
                 averaging_mode: Union[AveragingMode, None] = None, *,
                 name: str = 'accuracy', dtype: tf.DType = tf.float32
                 ):
        super().__init__(transform, averaging_mode, name=name, dtype=dtype)
        self._bifurcators: int = bifurcators
        self._include_endpoints: bool = include_endpoints
    
    def update(self, targets: TensorLike, scores: TensorLike,
               weights: TensorLike = tf.constant(1.0)) -> None:
        targets, predictions, weights \
            = self._preprocess(targets, scores, weights)
        
        confusion_matrix, _ = \
            compute_confusion_matrix(targets, predictions,
                                     bifurcators=self._bifurcators,
                                     weights=weights)
        
        tp = confusion_matrix['true_positives']
        fp = confusion_matrix['false_positives']
        
        if not self._include_endpoints:
            tp, fp = tp[1:-1], fp[1:-1]
        
        if self.average_mode == AveragingMode.SAMPLE:
            raise NotImplemented
        
        if self.average_mode == AveragingMode.MICRO:
            tp = tf.reduce_sum(tp, axis=-1)
            fp = tf.reduce_sum(fp, axis=-1)
        
        if self._value_sum is None or self._count is None:
            self._value_sum, self._count = self._create_variables(tf.shape(tp))
        
        self._value_sum.assign_add(tp)
        self._count.assign_add(tp + fp)


class RecallScore(ClassificationMetric):
    """
      Args:
      transform:          Transforms the prediction tensor; by default a sigmoid transforms is applied.

      bifurcators:         The number of equally distributed thresholds between 0 and 1; the default values is 1.
                           This means by default the metric is evaluated at a decision threshold of 0.5.

      include_endpoints:   Weather to include the endpoints, i.e. 0 and 1, as thresholds for precision evaluation.
                           By default, the endpoints are not included while reporting the precision scores.

      averaging_mode:     The  accumulating mode for the metric

                           'sample':     Calculate metrics for each instance, and report the average subset precision.
                                         (only meaningful for multilabel classification where this differs
                                         from accuracy_score).

                           'micro':      Calculates the precision over all prediction elements by inflating the
                                         prediction and label tensors.

                           'macro':      Calculates mean precision for each class and report their unweighted mean.
                                         This does not take label-imbalance into account.

                           'weighted':   Calculates mean accuracy for each class and reports a weighted, by support
                                         (the number of true instances for each label), average. This alters ‘macro’
                                         to account for label imbalance; it can result in an F-score that is not
                                         between precision and recall.

                           None:        Provides metrics for each class independently. By default, this metric
                                         returns the class-wise accuracy score.

              **kwargs                   Any additional arguments accepted by the base class

      name:               The name of the metric.
    """
    
    def __init__(self, transform: Callable = lambda x: x,
                 bifurcators: int = 1,
                 include_endpoints: bool = False,
                 averaging_mode: Union[AveragingMode, None] = None, *,
                 name: str = 'accuracy', dtype: tf.DType = tf.float32
                 ):
        super().__init__(transform, averaging_mode, name=name, dtype=dtype)
        self._bifurcators: int = bifurcators
        self._include_endpoints: bool = include_endpoints
    
    def update(self, targets: TensorLike, scores: TensorLike,
               weights: TensorLike = tf.constant(1.0)) -> None:
        targets, predictions, weights \
            = self._preprocess(targets, scores, weights)
        
        confusion_matrix, _ = \
            compute_confusion_matrix(targets, predictions,
                                     bifurcators=self._bifurcators,
                                     weights=weights)
        
        tp = confusion_matrix['true_positives']
        fn = confusion_matrix['false_negatives']
        
        if not self._include_endpoints:
            tp, fn = tp[1:-1], fn[1:-1]
        
        if self.average_mode == AveragingMode.SAMPLE:
            raise NotImplemented
        
        if self.average_mode == AveragingMode.MICRO:
            tp = tf.reduce_sum(tp, axis=-1)
            fn = tf.reduce_sum(fn, axis=-1)
        
        if self._value_sum is None or self._count is None:
            self._value_sum, self._count = self._create_variables(tf.shape(tp))
        
        self._value_sum.assign_add(tp)
        self._count.assign_add(tp + fn)


class AveragePrecisionScore(ClassificationMetric):
    """
      Args:
      transform:          Transforms the prediction tensor; by default a sigmoid transforms is applied.
      
      bifurcators:        The number of thresholds applied when calculating the score. The thresholds are evenly
                          distributed between the endpoints [0-1]

      averaging_mode:     The  accumulating mode for the metric


                         'sample':      Computes average subset accuracy at `sample` granularity. An average rate
                                         of  sample matches is reported, with a `match`  defined as all labels being
                                         correctly predicted. An average accuracy over all the samples is reported.

                         'micro':        Calculates the mean accuracy over all prediction elements by inflating the
                                         prediction and label tensors. An average accuracy over all predictions
                                         is reported.

                         'macro':        Calculates mean accuracy for each class and report their unweighted mean.
                                         This does not take label-imbalance into account.

                         'weighted':     Calculates mean accuracy for each class and reports a weighted, by support
                                         (the number of true instances for each label), average. This alters ‘macro’
                                         to account for label imbalance; it can result in an F-score that is not
                                         between precision and recall.

                         None:           Provides metrics for each class independently. By default, this metric
                                         returns the class-wise accuracy score.

      **kwargs           Any additional arguments accepted by the base class
              
      name:              The name of the metric.
    """
    
    def __init__(self, transform: Callable = lambda x: x,
                 bifurcators: int = 500,
                 averaging_mode: Union[AveragingMode, None] = None, *,
                 name: str = 'accuracy', dtype: tf.DType = tf.float32
                 ):
        super().__init__(transform, averaging_mode, name=name, dtype=dtype)
        
        averaging_mode = None \
            if averaging_mode in [AveragingMode.MACRO, AveragingMode.WEIGHTED]\
            else averaging_mode
        
        self._precision = PrecisionScore(bifurcators=bifurcators,
                                         include_endpoints=True,
                                         averaging_mode=averaging_mode)
        
        self._recall = RecallScore(bifurcators=bifurcators,
                                   include_endpoints=True,
                                   averaging_mode=averaging_mode)
    
    def update(self, targets: TensorLike, scores: TensorLike,
               weights: TensorLike = tf.constant(1.0)) -> None:
    
        targets, predictions, weights \
            = self._preprocess(targets, scores, weights)
        
        self._precision.update(targets, predictions, weights)
        self._recall.update(targets, predictions, weights)
        
        if self._count is None:
            _, self._count = self._create_variables(targets.shape[-1])
        self._count.assign_add(tf.reduce_sum(targets * weights, axis=0))
    
    def result(self) -> TensorLike:
        avg_precision = -tf.reduce_sum(tnp.diff(self._recall.result(), axis=0)
                                       * self._precision.result()[:-1], axis=0)
        
        if self.average_mode in [AveragingMode.MACRO, AveragingMode.WEIGHTED]:
            if self.average_mode == AveragingMode.WEIGHTED:
                weights = tf.math.divide_no_nan(self._count, tf.reduce_sum(self._count))
                avg_precision = tf.reduce_sum(weights * avg_precision, axis=-1)
            else:
                avg_precision = tf.reduce_mean(avg_precision, axis=-1)
        return avg_precision
    
    def reset(self) -> None:
        self._precision.reset()
        self._recall.reset()
        
        
class AccuracyScore(ClassificationMetric):
    """
      Args:
      transform:          Transforms the prediction tensor; by default a sigmoid transforms is applied.
      
      subset_mode:        When this boolean flag is set `sample` averaging mode computes a subset accuracy.
                          This flag does not affect other averaging mode. See the averaging_mode option for more
                          details.

      averaging_mode:     The  accumulating mode for the metric

                          'sample':      When `subset_mode` is set, this computes subset accuracy at sample granularity.
                                         An average rate of  sample matches is reported, with a `match` defined as all
                                         labels being correctly predicted. When `subset_mode` is False, this computes
                                         the (TP+TN)/total for each sample and reports the average over all samples.

                          'micro':       Calculates the mean accuracy over all prediction elements by inflating the
                                         prediction and label tensors. An average accuracy over all predictions
                                         is reported.

                          'macro':       Calculates mean accuracy for each class and report their unweighted mean.
                                         This does not take label-imbalance into account.

                          'weighted':    Calculates mean accuracy for each class and reports a weighted, by support
                                         (the number of true instances for each label), average. This alters ‘macro’
                                         to account for label imbalance; it can result in an F-score that is not
                                         between precision and recall.

                          None:         Provides metrics for each class independently. By default, this metric
                                         returns the class-wise accuracy score.

      **kwargs           Any additional arguments accepted by the base class
              
      name:               The name of the metric.
    """
    
    def __init__(self, transform: Callable = lambda x: x, subset_mode: bool = True,
                 averaging_mode: Union[AveragingMode, None] = None, *,
                 name: str = 'accuracy', dtype: tf.DType = tf.float32
                 ):
        self._subset_mode = subset_mode
        super().__init__(transform, averaging_mode, name=name, dtype=dtype)
    
    def update(self, targets: TensorLike, scores: TensorLike,
               weights: TensorLike = tf.constant(1.0)) -> None:
        """
        
        Args:
            targets:     A tensor of shape (n_samples, n_classes) or (n_samples,)
                        Ground truth (correct) target values.
                        
            scores:     A tensor of shape (n_samples, n_classes) or (n_samples,)
                        Estimated targets as returned by a classifier.
                        
            weights:    A tensor of shape (n_samples,) or any shape that can broadcast to y_pred/y_true.
                        The default value is == `1.0`, which implies weights are distributed equally.
                        
                        This argument can be used to assign class weights as well.To do so a weight tensor of shape
                        (1, n_classes) is passed, which effectively assigns each class a corresponding weight and
                        broadcast it to all samples. To assign different class weight to different samples a weight
                        tensor of shape (n_samples,n_classes) must be passed.
  
        Returns:
                       None
        """
        targets, predictions, weights \
            = self._preprocess(targets, scores, weights)
        
        if self.average_mode == AveragingMode.SAMPLE:
         
            if self._subset_mode:
                reduce_op = compose(partial(tf.reduce_all, keepdims=True, axis=-1),
                                    partial(tf.cast, dtype=self.dtype)
                                    )
            else:
                reduce_op = compose(partial(tf.cast, dtype=self.dtype),
                                    partial(tf.reduce_mean, keepdims=True, axis=-1))
            #TODO: verify spreading weights among positive labels
            # Further tests needed for this mode.
            label_weights = (targets * weights) / tf.reduce_sum(targets, axis=-1, keepdims=1)
            hits = reduce_op(predictions == targets)
            weighted_hits = label_weights * hits
            value_sum = tf.cast(tf.reduce_sum(weighted_hits), self.dtype)
            count = tf.cast(tf.reduce_sum(label_weights), self.dtype)

            if self._value_sum is None or self._count is None:
                self._value_sum, self._count = \
                    self._create_variables(tuple())
            self._value_sum.assign_add(value_sum)
            self._count.assign_add(count)
        
        elif self.average_mode == AveragingMode.MICRO:

            label_weights = tf.ones_like(targets) * weights
            label_weights = tf.reshape(label_weights, (-1))
            targets = tf.reshape(targets, (-1, 1))
            predictions = tf.reshape(predictions, (-1, 1))
            hits = tf.cast(tf.reduce_all(predictions == targets, axis=-1), self.dtype)
            weighted_hits = label_weights * hits
            value_sum = tf.cast(tf.reduce_sum(weighted_hits), self.dtype)
            count = tf.cast(tf.reduce_sum(label_weights), self.dtype)
        
            if self._value_sum is None or self._count is None:
                self._value_sum, self._count = \
                    self._create_variables(tuple())
            self._value_sum.assign_add(value_sum)
            self._count.assign_add(count)
        
        else:
            confusion_matrix, _ = compute_confusion_matrix(targets, predictions,
                                                           bifurcators=1,
                                                           weights=weights)
            tp = confusion_matrix['true_positives'][1:-1]
            tn = confusion_matrix['true_negatives'][1:-1]
            fp = confusion_matrix['false_positives'][1:-1]
            fn = confusion_matrix['false_negatives'][1:-1]
            
            if self._value_sum is None or self._count is None:
                self._value_sum, self._count = \
                    self._create_variables(tf.shape(tp))
            
            # accuracy = TP+TN/(TP+TN+FN_+FP)
            self._value_sum.assign_add(tp + tn)
            self._count.assign_add(tp + tn + fn + fp)