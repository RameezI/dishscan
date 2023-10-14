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

import math
import tensorflow as tf
from typing import Optional
from .utils import capture_params


class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule.
    """
    
    @capture_params
    def __init__(self, initial_learning_rate: float, warmup_steps: int, decay_steps: int,
                 alpha: float = 0.0, name: Optional[str] = None,
                 dtype: tf.DType = tf.float32):
        
        super().__init__()
        
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=dtype,
                                                          name="initial_learning_rate")
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.decay_steps = tf.cast(decay_steps, dtype)
        self.alpha = tf.cast(alpha, dtype)
        self.name = name
        self.dtype = dtype
    
    @tf.function
    def __call__(self, step: tf.constant):
        
        global_step = tf.cast(step, self.dtype)
        
        if global_step <= self.warmup_steps:
            linear_ramp = global_step / self.warmup_steps
            factor = (1 - self.alpha) * linear_ramp + self.alpha
        
        else:
            decay_step = tf.minimum(global_step-self.warmup_steps, self.decay_steps)
            progress = decay_step / self.decay_steps
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=self.dtype) * progress))
            factor = (1 - self.alpha) * cosine_decayed + self.alpha
        
        return tf.multiply(self.initial_learning_rate, factor)
    
    def get_config(self):
        params = getattr(self, 'params', {})
        return params