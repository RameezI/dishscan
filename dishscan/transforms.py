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
from typing import Union, Tuple
import tensorflow_addons as tfa
from .utils import capture_params


# Aliases
ResizeMethod = tf.image.ResizeMethod


class Shuffle:
    @capture_params
    def __init__(self, buffer_size: int = 2048, seed=None,
                 reshuffle_each_iteration=False):
        self.buffer_size = buffer_size
        self.seed = None
        self.reshuffle_each_iteration = reshuffle_each_iteration
    
    def __call__(self, dataset):
        return dataset.shuffle(self.buffer_size, seed=self.seed,
                               reshuffle_each_iteration=self.reshuffle_each_iteration)


class Cache:
    def __call__(self, dataset):
        return dataset.cache()


class Prefetch:
    @capture_params
    def __init__(self, buffer_len=tf.data.AUTOTUNE):
        self.buffer_len = buffer_len
    
    def __call__(self, dataset):
        return dataset.prefetch(self.buffer_len)


class Normalize:
    @capture_params
    def __init__(self, mean=0.0, std=1.0, keys=('image',),
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.mean = tf.constant(mean)
        self.std = tf.constant(std)
        self.keys = keys
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        for key in self.keys:
            batch.update({key: (tf.cast(batch[key], tf.float32) - self.mean) / self.std})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class DropRemainder:
    @capture_params
    def __init__(self, batch_size, key='image', batch_first=True) -> None:
        self.key = key
        self.batch_size = batch_size
        self.batch_first = batch_first
    
    def filter(self, batch):
        axis = 0 if self.batch_first else -1
        is_proper_size = tf.math.equal(tf.shape(batch[self.key])[axis], self.batch_size)
        return is_proper_size
    
    def __call__(self, dataset):
        return dataset.filter(self.filter)


class TakeSingleImage:
    @capture_params
    def __init__(self, index: Union[int, None] = None,
                 input_key: str = 'video',
                 output_key: str = 'image',
                 drop_input_key: bool = True,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 **kwargs) -> None:
        """
        Extracts a single image, from a sequence of images, and adds it as an additional feature to the batch
        Args:
            index:              The index of the sequence, if None a random index is generated.
            input_key:          The key under which the sequence of images is available
            output_key:         The key under which the extracted image is to be saved
            drop_input_key:     Weather to drop the original feature from which the image is extracted
            **kwargs            additional arguments
        """
        self.image_index = index
        self.input_key = input_key
        self.output_key = output_key
        self.drop_input_key = drop_input_key
        self.index_limit = kwargs.pop('index_limit', None)
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        seq_len = int(tf.shape(batch[self.input_key])[1]) \
            if self.index_limit is None else self.index_limit
        index = self.image_index if self.image_index is not None \
            else tf.random.uniform((), minval=0, maxval=seq_len, dtype=tf.int32)
        batch[self.output_key] = batch[self.input_key][:, index, :, :, :]
        batch.pop('video') if self.drop_input_key else None
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class OneHotLabels:
    @capture_params
    def __init__(self, num_labels=10, keys=('label',),
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.keys = keys
        self.num_labels = num_labels
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        for key in self.keys:
            batch.update({key: tf.one_hot(tf.cast(batch[key], tf.int32), self.num_labels)})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class SupervisedSignal:
    @capture_params
    def __init__(self, keys=('image', 'label'),
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.supervised_keys = keys
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        batch = batch[self.supervised_keys[0]], batch[self.supervised_keys[1]]
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class Resize:
    @capture_params
    def __init__(self, size, input_key='image', method: ResizeMethod = ResizeMethod.BILINEAR,
                 preserve_aspect_ratio: bool = False, antialias: bool = False,
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.key = input_key
        self.size = size
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.antialias = antialias
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        out_images = tf.image.resize(batch[self.key], self.size,
                                     method=self.method,
                                     preserve_aspect_ratio=self.preserve_aspect_ratio,
                                     antialias=self.antialias)
        batch.update({self.key: out_images})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class RandomCropWithPadding:
    @capture_params
    def __init__(self, image_size: Union[int, Tuple[int, int]],
                 padding_size: Union[int, Tuple[int, int]] = 0,
                 input_key='image',
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        """
        Args:
            image_size:
            padding_size:
            input_key:
        """
        
        self.key = input_key
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(padding_size, int):
            padding_size = (padding_size, padding_size)
        
        if len(padding_size) == 2:
            self.pad_rows = padding_size[0]
            self.pad_cols = padding_size[1]
        else:
            raise ValueError
        
        if len(image_size) == 2:
            self.image_height = image_size[0]
            self.image_width = image_size[1]
        else:
            raise ValueError
        
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        padded_height = self.image_height + int(2 * self.pad_rows)
        padded_width = self.image_width + int(2 * self.pad_cols)
        padded_image = tf.image.pad_to_bounding_box(batch[self.key],
                                                    self.pad_rows, self.pad_cols,
                                                    padded_height, padded_width)
        y = tf.random.uniform(shape=[], minval=0, maxval=self.pad_rows,
                              dtype=tf.int32)
        x = tf.random.uniform(shape=[], minval=0, maxval=self.pad_cols,
                              dtype=tf.int32)
        output_image = tf.image.crop_to_bounding_box(padded_image, y, x,
                                                     self.image_height,
                                                     self.image_width)
        batch.update({self.key: output_image})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class RandomHorizontalFlip:
    @capture_params
    def __init__(self, input_key='image',
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.key = input_key
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        output_image = tf.image.random_flip_left_right(batch[self.key])
        batch.update({self.key: output_image})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class RandomVerticalFlip:
    @capture_params
    def __init__(self, input_key='image',
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.key = input_key
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        output_image = tf.image.random_flip_up_down(batch[self.key])
        batch.update({self.key: output_image})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, num_parallel_calls=self.num_parallel_calls)


class RandomCutout:
    @capture_params
    def __init__(self, cutout_size, input_key='image',
                 num_parallel_calls=tf.data.AUTOTUNE) -> None:
        self.key = input_key
        self.cutout_size = cutout_size
        self.num_parallel_calls = num_parallel_calls
    
    def transform(self, batch):
        output_image = tfa.image.random_cutout(batch[self.key], self.cutout_size)
        batch.update({self.key: output_image})
        return batch
    
    def __call__(self, dataset):
        return dataset.map(self.transform, self.num_parallel_calls)

# @tf.function
# def _norm_params(mask_size, offset=None):
#     tf.assert_equal(
#         tf.reduce_any(mask_size % 2 != 0),
#         False,
#         "mask_size should be divisible by 2",
#     )
#     if tf.rank(mask_size) == 0:
#         mask_size = tf.stack([mask_size, mask_size])
#     if offset is not None and tf.rank(offset) == 1:
#         offset = tf.expand_dims(offset, 0)
#     return mask_size, offset
#
#
# @tf.function
# def _random_center(mask_dim_length, image_dim_length, batch_size, seed):
#     if mask_dim_length >= image_dim_length:
#         return tf.tile([image_dim_length // 2], [batch_size])
#     half_mask_dim_length = mask_dim_length // 2
#     return tf.random.uniform(
#         shape=[batch_size],
#         minval=half_mask_dim_length,
#         maxval=image_dim_length - half_mask_dim_length,
#         dtype=tf.int32,
#         seed=seed,
#     )
#
#
# def cutout(images: TensorLike, mask_size: TensorLike,
#            offset: TensorLike = (0, 0), constant_values: Number = 0,) -> tf.Tensor:
#     """Apply [cutout](https://arxiv.org/abs/1708.04552) to images.
#     This operation applies a `(mask_height x mask_width)` mask of zeros to
#     a location within `images` specified by the offset.
#     The pixel values filled in will be of the value `constant_values`.
#     The location where the mask will be applied is randomly
#     chosen uniformly over the whole images.
#     Args:
#       images: A tensor of shape `(batch_size, height, width, channels)`.
#       mask_size: Specifies how big the zero mask that will be generated is that
#         is applied to the images. The mask will be of size
#         `(mask_height x mask_width)`. Note: mask_size should be divisible by 2.
#       offset: A tuple of `(height, width)` or `(batch_size, 2)`
#       constant_values: What pixel value to fill in the images in the area that has
#         the cutout mask applied to it.
#     Returns:
#       A `Tensor` of the same shape and dtype as `images`.
#     Raises:
#       InvalidArgumentError: if `mask_size` can't be divisible by 2.
#     """
#     with tf.name_scope("cutout"):
#         images = tf.convert_to_tensor(images)
#         mask_size = tf.convert_to_tensor(mask_size)
#         offset = tf.convert_to_tensor(offset)
#
#         image_static_shape = images.shape
#         image_dynamic_shape = tf.shape(images)
#         image_height, image_width, channels = (
#             image_dynamic_shape[1],
#             image_dynamic_shape[2],
#             image_dynamic_shape[3],
#         )
#
#         mask_size, offset = _norm_params(mask_size, offset)
#         mask_size = mask_size // 2
#
#         cutout_center_heights = offset[:, 0]
#         cutout_center_widths = offset[:, 1]
#
#         lower_pads = tf.maximum(0, cutout_center_heights - mask_size[0])
#         upper_pads = tf.maximum(0, image_height - cutout_center_heights - mask_size[0])
#         left_pads = tf.maximum(0, cutout_center_widths - mask_size[1])
#         right_pads = tf.maximum(0, image_width - cutout_center_widths - mask_size[1])
#
#         cutout_shape = tf.transpose(
#             [
#                 image_height - (lower_pads + upper_pads),
#                 image_width - (left_pads + right_pads),
#             ],
#             [1, 0],
#         )
#
#         def fn(i):
#             padding_dims = [
#                 [lower_pads[i], upper_pads[i]],
#                 [left_pads[i], right_pads[i]],
#             ]
#             mask = tf.pad(
#                 tf.zeros(cutout_shape[i], dtype=tf.bool),
#                 padding_dims,
#                 constant_values=True,
#             )
#             return mask
#
#         mask = tf.map_fn(
#             fn,
#             tf.range(tf.shape(cutout_shape)[0]),
#             fn_output_signature=tf.TensorSpec(
#                 shape=image_static_shape[1:-1], dtype=tf.bool
#             ),
#         )
#         mask = tf.expand_dims(mask, -1)
#         mask = tf.tile(mask, [1, 1, 1, channels])
#
#         images = tf.where(
#             mask,
#             images,
#             tf.cast(constant_values, dtype=images.dtype),
#         )
#         images.set_shape(image_static_shape)
#         return images
#
#
# def random_cutout(images: TensorLike, mask_size: TensorLike, constant_values: Number = 0.) -> tf.Tensor:
#     """Apply [cutout](https://arxiv.org/abs/1708.04552) to images with random offset.
#     This operation applies a `(mask_height x mask_width)` mask of zeros to
#     a random location within `images`. The pixel values filled in will be of
#     the value `constant_values`. The location where the mask will be applied is
#     randomly chosen uniformly over the whole images.
#
#     Args:
#       images:           A tensor of shape `(batch_size, height, width, channels)`.
#       mask_size:        Specifies how big the zero mask that will be generated is that
#                         is applied to the images. The mask will be of size
#                         `(mask_height x mask_width)`. Note: mask_size should be divisible by 2.
#       constant_values:  What pixel value to fill in the images in the area that has
#                         the cutout mask applied to it.
#
#     Returns:
#       A `Tensor` of the same shape and dtype as `images`
#     """
#     images = tf.convert_to_tensor(images)
#     mask_size = tf.convert_to_tensor(mask_size)
#
#     image_dynamic_shape = tf.shape(images)
#     batch_size, image_height, image_width = (
#         image_dynamic_shape[0],
#         image_dynamic_shape[1],
#         image_dynamic_shape[2],
#     )
#
#     mask_size, _ = _norm_params(mask_size, offset=None)
#
#     cutout_center_height = _random_center(mask_size[0], image_height, batch_size, seed)
#     cutout_center_width = _random_center(mask_size[1], image_width, batch_size, seed)
#
#     offset = tf.transpose([cutout_center_height, cutout_center_width], [1, 0])
#     return cutout(images, mask_size, offset, constant_values)