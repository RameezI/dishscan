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

"""nutrition5k dataset."""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

_DESCRIPTION = """
Nutrition5k is a dataset of visual and nutritional data for ~5k realistic plates of food, originally released
by Google, captured from Google cafeterias using a custom scanning rig. The dataset is published alongside a
CVPR 2021 paper with the aim to promote research in visual nutrition understanding.
The dishes in Nutrition5k vary drastically in portion sizes and dish complexity, with dishes ranging from just
a few calories to over 1000 calories and from a single ingredient to up to 35 with an average of 5.7 ingredients
per plate.(https://github.com/google-research-datasets/Nutrition5k)

This version of the dataset is compiled to build upon the current stat-of-the-art and take on the challenge of
automatic visual estimation of macro-nutrients, in order in order to promote healthy living.

Author: rameez.ismaeel@gmail.com
"""

_CITATION = """
@inproceedings{thames2021nutrition5k
  title={Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food},
  author={Thames, Quin and Karpur, Arjun and Norris, Wade and Xia, Fangting and Panait, Liviu and Weyand,
          Tobias and Sim, Jack},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8903--8911},
  year={2021}
}
"""
_IGNORE_IDS =['dish_1560368815_D']
_URL_BASE = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset.tar.gz'
#_MACROS = ('carb', 'fat', 'protein')
_VALID_RGB_VARIANTS = ['res448', 'res224', 'res112']

INGREDIENTS = pd.read_csv(Path(os.path.dirname(__file__), 'ingredients_metadata.csv'),
                          index_col='id')['ingr'].to_dict()


class Nutrition5kConfig(tfds.core.BuilderConfig):
    def __init__(self, resolution=(448, 448), channels=3, **kwargs):
        if kwargs.get('name', '??') not in _VALID_RGB_VARIANTS:
            raise ValueError('Value Error! Cannot identify the requested variant.\n'
                             f'The variant name must be one of {_VALID_RGB_VARIANTS}\n'
                             f"requested: {kwargs.get('name', '?')}")
        super(Nutrition5kConfig, self).__init__(**kwargs)
        self.resolution = resolution
        self.channels = channels


class Nutrition5k(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for nutrition5k dataset."""
    
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
     If this script fails to (automatically) download the dataset please manually download it from this
     [download_link](https://storage.cloud.google.com/nutrition5k_dataset/nutrition5k_dataset.tar.gz)\n

     To build the dataset from this manual download you must specify the directory containing the
     nutrition5k_dataset.tar.gz through the `manual_dir` option, defaults to:
      ~/tensorflow_datasets/downloads/manual.

     For example to build this dataset using CLI:
        ```
        tfds build --data_dir <tensorflow_datasets> --manual_dir <directory_containing_nutrition5k_dataset.tar.gz>
        ```
     """
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    VIDEO_RGB_DESC = """"
    The `video_rgb` variant of the dataset constructs an input pipeline using the video sequence recorded
    by the side angle cameras. Four such cameras are used to capture' the side angles of a plate, thus
    producing four independent examples of the every food plate. Each video is around 8 sec long."""
    
    BUILDER_CONFIGS = [
        Nutrition5kConfig(
            name='res448',
            description=VIDEO_RGB_DESC,
            resolution=(448, 448),  # height, width
            channels=3,
        ),
        Nutrition5kConfig(
            name='res224',
            description=VIDEO_RGB_DESC,
            resolution=(224, 224),  # height, width
            channels=3,
        ),
        Nutrition5kConfig(
            name='res112',
            description=VIDEO_RGB_DESC,
            resolution=(112, 112),  # height, width
            channels=3,
        )
    ]
    
    def _info(self) -> tfds.core.DatasetInfo:
        shape_image = self.builder_config.resolution + (self.builder_config.channels,)
        shape_video = (None,) + shape_image
        shape_ingredients = (len(INGREDIENTS),)
        #shape_macros = (len(INGREDIENTS), len(_MACROS))
        
        h, w = self.builder_config.resolution
        #1 graph_vf = r"crop=in_w/1.2:in_h/1.2, mpdecimate=hi=64*128:lo=64*64:frac=0.5"
        #2 # graph_vf = r"mpdecimate=hi=64*128:lo=64*64:frac=0.5"
        graph_vf = "select=eq(n\,0)"
        args_ffmpeg = ['-s', f'{w}x{h}', '-filter:v', graph_vf, '-an',
                       '-fps_mode', 'passthrough']
        
        """Returns the dataset metadata."""
        if self.builder_config.name in _VALID_RGB_VARIANTS:
            features = tfds.features.FeaturesDict({'id': tfds.features.Text(),
                                                   'video': tfds.features.Video(shape=shape_video,
                                                                                ffmpeg_extra_args=args_ffmpeg),
                                                   'ingredients': tfds.features.Tensor(shape=shape_ingredients,
                                                                                       dtype=tf.float32),
                                                   # 'weight/ingredient': tfds.features.Tensor(shape=shape_ingredients,
                                                   #                                           dtype=tf.float32),
                                                   # 'calories/ingredient': tfds.features.Tensor(shape=shape_ingredients,
                                                   #                                             dtype=tf.float32),
                                                   # 'macros/ingredient': tfds.features.Tensor(shape=shape_macros,
                                                   #                                           dtype=tf.float32),
                                                   })
        else:
            raise ValueError('Value Error! Cannot identify the requested variant.\n'
                             f'The variant name must be one of {_VALID_RGB_VARIANTS}\n'
                             f'requested: {self.builder_config.name}')
        
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage='https://github.com/google-research-datasets/Nutrition5k',
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # try:
        extract_path = dl_manager.download_and_extract(_URL_BASE)
        # except Exception as e:
        #     print(f'Automatic download failure\n {e}\n\n'
        #           f'attempting to build from a manual location: {dl_manager.manual_dir}')
        #     download_name = 'nutrition5k_dataset.tar.gz'
        #     extract_path = dl_manager.extract(Path(dl_manager.manual_dir, download_name))
        
        extract_path = Path(extract_path, 'nutrition5k_dataset')
        splits_dir = Path(extract_path, 'dish_ids', 'splits')
        
        if self.builder_config.name in _VALID_RGB_VARIANTS:
            train_file = os.path.join(splits_dir, 'rgb_train_ids.txt')
            test_file = os.path.join(splits_dir, 'rgb_test_ids.txt')
    
        else:
            raise ValueError('Value Error! Cannot identify the requested variant.\n'
                             f'The variant name must be one of {_VALID_RGB_VARIANTS}\n'
                             f'requested: {self.builder_config.name}')
        
        with tf.io.gfile.GFile(train_file) as file:
            train_keys = file.readlines()
            train_keys = [key.rstrip() for key in train_keys]
        
        with tf.io.gfile.GFile(test_file) as file:
            test_keys = file.readlines()
            test_keys = [key.rstrip() for key in test_keys]
        
        # self.extract_frames(extract_path)
        return {
            'train': self._generate_examples(extract_path, keys=train_keys),
            'test': self._generate_examples(extract_path, keys=test_keys)
        }
    
    def _generate_examples(self, path, keys):
        """Yields examples."""
        
        metadata_stem = Path(path, 'metadata')
        metadata_files = [file for file in metadata_stem.glob('*.csv')]
        meta_columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
        col_lists = [[f'ingredient_{k}', f'ingredient_{k}_name', f'ingredient_{k}_grams',
                      f'ingredient_{k}_calories', f'ingredient_{k}_fat',
                      f'ingredient_{k}_carb', f'ingredient_{k}_protein']
                     for k in range(len(INGREDIENTS))]
        
        ingredient_columns = [col for col_list in col_lists for col in col_list]
        meta_columns.extend(ingredient_columns)
        
        # read metadata
        reader_config = {'header': None, 'names': meta_columns, 'index_col': 'dish_id'}
        metadata = pd.concat([pd.read_csv(meta_file, **reader_config)
                              for meta_file in metadata_files], axis=0)
        
        # self.extract_frames(path)
        for key in keys:
            cols_id_ingredient = [f'ingredient_{k}' for k in range(len(INGREDIENTS))]
            cols_weight_ingredient = [f'ingredient_{k}_grams' for k in range(len(INGREDIENTS))]

            #cols_carb_ingredient = [f'ingredient_{k}_carb' for k in range(len(INGREDIENTS))]
            #cols_fat_ingredient = [f'ingredient_{k}_fat' for k in range(len(INGREDIENTS))]
            #cols_protein_ingredient = [f'ingredient_{k}_protein' for k in range(len(INGREDIENTS))]
            #cols_calories_ingredient = [f'ingredient_{k}_calories' for k in range(len(INGREDIENTS))]
            
            id_ingredients = metadata.loc[key, cols_id_ingredient].dropna()
            id_ingredients = id_ingredients.map(lambda x: int(x.split('_')[1]))
            weight_ingredients = metadata.loc[key, cols_weight_ingredient].dropna()
            
            #calories_ingredients = metadata.loc[key, cols_calories_ingredient].dropna()
            #carb_ingredients = metadata.loc[key, cols_carb_ingredient].dropna()
            #fat_ingredients = metadata.loc[key, cols_fat_ingredient].dropna()
            #protein_ingredients = metadata.loc[key, cols_protein_ingredient].dropna()
            
            # sanity checks
            assert id_ingredients.is_unique, 'The ingredients must be unique, repetition is not allowed!'
            assert all(id_ingredients.between(1, len(INGREDIENTS))), \
                f'The ingredients_id must be in the range:  {1}-{len(INGREDIENTS)}\n' \
                'The above constraint is not respected!'
            
            if (weight_ingredients > 2000).any():
                # if any ingredient has weight greater than 2kg; skip to the next key!
                continue
            
            ingredients_indices = [int(ingredient_id - 1)
                                   for ingredient_id in id_ingredients]
            
            # multiple-hot labels
            one_hot_labels = tf.one_hot(ingredients_indices, depth=len(INGREDIENTS))
            ingredients = tf.reduce_sum(one_hot_labels, axis=0)
            
            # weights
            ingredients_indices = tf.expand_dims(ingredients_indices, axis=-1)
            weights = weight_ingredients.astype(np.float32).tolist()
            weight_ingredients = tf.scatter_nd(ingredients_indices, weights, [len(INGREDIENTS)])
            
            # calories
            #calories = calories_ingredients.astype(np.float32).tolist()
            #calories_ingredients = tf.scatter_nd(ingredients_indices, calories, [len(INGREDIENTS)])
            
            # macros
            # macros = list(zip(carb_ingredients.astype(np.float32).tolist(),
            #                   fat_ingredients.astype(np.float32).tolist(),
            #                   protein_ingredients.astype(np.float32).tolist()))
            # macros_ingredients = tf.scatter_nd(ingredients_indices, macros, [len(INGREDIENTS), 3])
            
            if self.builder_config.name in _VALID_RGB_VARIANTS:
                data_stem = Path(path, 'imagery', 'side_angles', key)
                data = {f'{key}_{str(file)[-6:-5]}': file
                        for file in data_stem.glob('*[ABCD].h264')}
                
                # encode each example
                for data_id, data_file in data.items():    
                    if data_id in _IGNORE_IDS:
                        # the video files with these ids cause issue with the
                        #  ffmpeg pipeline, skipping for now.
                        continue

                    yield data_id, {'id': data_id,
                                    'video': tf.io.gfile.GFile(data_file, 'rb'),
                                    'ingredients': np.array(ingredients),
                                    #'weight/ingredient': np.array(weight_ingredients),
                                    #'calories/ingredient': np.array(calories_ingredients),
                                    #'macros/ingredient': np.array(macros_ingredients),
                                    }
            else:
                raise ValueError('Value Error! Cannot identify the requested variant.\n'
                                 f'The variant name must be one of {_VALID_RGB_VARIANTS}\n'
                                 f'requested: {self.builder_config.name}')