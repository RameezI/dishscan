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
from dishscan.utils import capture_params
from typing import Union, Callable


class FeatureExtractor(tf.keras.Model):
    
    def get_config(self):
        return getattr(self, 'params', dict())
    
    def __init__(self, pre_processor: Union[Callable, None], model: tf.keras.Model,
                 trainable: bool = True) -> None:
        super().__init__()
        self.preprocessor = pre_processor
        self._model = model
        
        self._model.trainable = trainable
        for layer in self._model.layers:
            layer.trainable = trainable
    
    def call(self, x, training=True):
        x = self.preprocessor(tf.cast(x, tf.float32)) if self.preprocessor else x
        training = training if self._model.trainable else False
        x = self._model(x, training=training)
        return x
    
    def to_json(self):
        params = getattr(self, 'params', {})
        return {'module': self.__module__,
                'class': type(self).__name__,
                'config': dict(params)}


class DenseNet121(FeatureExtractor):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.densenet.preprocess_input \
            if pre_process else None
        model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                           weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class DenseNet169(FeatureExtractor):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.densenet.preprocess_input \
            if pre_process else None
        model = tf.keras.applications.densenet.DenseNet169(include_top=False,
                                                           weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class DenseNet201(FeatureExtractor):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.densenet.preprocess_input \
            if pre_process else None
        model = tf.keras.applications.densenet.DenseNet201(include_top=False,
                                                           weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class MobileNetV1(FeatureExtractor):

    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):

        preprocessor = tf.keras.applications.mobilenet.preprocess_input \
            if pre_process else None

        model = tf.keras.applications.MobileNet(include_top=False,
                                                weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class MobileNetV2(FeatureExtractor):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.mobilenet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.MobileNetV2(include_top=False,
                                                  weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class Xception(FeatureExtractor):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.xception.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.Xception(include_top=False,
                                               weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class InceptionV3(FeatureExtractor):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.inception_v3.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.InceptionV3(include_top=False,
                                                  weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB0(FeatureExtractor):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                     weights=weights)
        
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB1(FeatureExtractor):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB1(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB2(FeatureExtractor):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB2(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB3(FeatureExtractor):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB3(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB4(FeatureExtractor):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB4(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)