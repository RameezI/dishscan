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

import re
import os
import numpy as np
from typing import Union, Dict
from datetime import datetime as dt
from datetime import timedelta
from enum import Enum


class ModeProgressBar(Enum):
    DEFAULT = 0
    EVAL_ONLY = 1


class ProgressBar:
    _proto_train = 'Epoch: {epoch:<{digits_epoch}} ' \
                   '[{percent_complete:>3}% == {samples_done:>{digits_samples}}' \
                   ' samples - {time_span}]'
    
    _proto_eval = 'Eval: [{samples_done:>{digits_samples}}' \
                  ' samples - {time_span}]'
    
    _proto_ckpt = '{ckpt}'
    
    _nan_pattern = re.compile('nan')
    
    def __init__(self):
        self.epoch: Union[int, np.NaN] = np.nan
        self.max_epochs: Union[int, np.NaN] = np.nan
        self.epoch_start_time: Union[dt.now, None] = None
        self.time_after_first_batch: Union[dt.now, None] = None
        self.eval_start_time: Union[dt.now, None] = None
        self.train_samples = np.nan
        self.eval_samples = np.nan
        
        self._train_metrics: Union[str, None] = None
        self._eval_metrics: Union[str, None] = None
        self._stream_out: Union[str, None] = None
        
        self._step_count = 0
        self._train_samples_done = 0
        self._eval_samples_done = 0
        self._percentage_complete = 0
        self._checkpoint: str = ''
        
        self._time_spent_train = self.get_epoch_time(dt.now())
        self._time_spent_eval = self.get_epoch_time(dt.now())
        
        self._mode: ModeProgressBar = ModeProgressBar.DEFAULT
        self._statement_train: str = self._proto_train.format(epoch=self.epoch,
                                                              samples_done=self._train_samples_done,
                                                              percent_complete=self.percent_complete,
                                                              time_span=self._time_spent_train,
                                                              digits_epoch=self.digits_epoch,
                                                              digits_samples=self.digits_train_samples,
                                                              )
        
        self._statement_eval: str = self._proto_eval.format(samples_done=self._eval_samples_done,
                                                            digits_samples=self.digits_eval_samples,
                                                            time_span=self._time_spent_eval)
        
        self._statement_ckpt: str = self._proto_ckpt.format(ckpt=self._checkpoint)
    
    @property
    def mode(self) -> ModeProgressBar:
        return self._mode
    
    @mode.setter
    def mode(self, value: ModeProgressBar):
        self._mode = value
    
    @property
    def percent_complete(self):
        return np.nan \
            if any(np.isnan([self.train_samples, self._train_samples_done])) else \
            int(min(self._train_samples_done * 100.0 / self.train_samples, 100.0))
    
    @property
    def digits_eval_samples(self):
        return len(str(100)) if np.isnan(self.eval_samples) \
            else len(str(self.eval_samples))
    
    @property
    def digits_train_samples(self):
        return len(str(100)) if np.isnan(self.train_samples) \
            else len(str(self.train_samples))
    
    @property
    def digits_epoch(self):
        return len(str(1000)) if np.isnan(self.max_epochs) \
            else len(str(self.max_epochs))
    
    @property
    def train_metrics(self):
        return self._train_metrics
    
    @train_metrics.setter
    def train_metrics(self, value):
        self._train_metrics = self.metrics_formatter(value)
    
    @property
    def eval_metrics(self):
        return self._eval_metrics
    
    @eval_metrics.setter
    def eval_metrics(self, value):
        self._eval_metrics = self.metrics_formatter(value)
    
    def get_step_time(self, step_count: int, now: dt.now):
        time_per_step: Union[float, timedelta] = np.NaN
        if step_count > 1 and self.time_after_first_batch is not None:
            delta = timedelta.total_seconds((now - self.time_after_first_batch))
            time_per_step = delta / (step_count - 1)
        if step_count > 0 and self.epoch_start_time is not None:
            delta = timedelta.total_seconds((now - self.epoch_start_time))
            time_per_step = delta / step_count
        return time_per_step
    
    def get_epoch_time(self, now: dt.now):
        delta: Union[float, timedelta] = np.NaN
        if self.epoch_start_time is not None:
            delta = now - self.epoch_start_time
        delta_str = str(delta).split('.')[0]
        return delta_str
    
    def get_eval_time(self, now: dt.now):
        delta: Union[float, timedelta] = np.NaN
        if self.eval_start_time is not None:
            delta = now - self.eval_start_time
        delta_str = str(delta).split('.')[0]
        return delta_str
    
    @staticmethod
    def metrics_formatter(metrics, separator=' | '):
        formatted_metrics = ''
        if metrics:
            formatted_metrics = [f'{key}: {value:<6.3f}' if isinstance(value, float) else
                                 f'{key}: {value:<6d}' for key, value in metrics.items()]
            formatted_metrics = separator.join(formatted_metrics)
        return formatted_metrics
    
    @property
    def train_statement(self):
        """  Updates the underlying statement, reflecting the current state of the
         instance variables, and returns the statement as formatted string"""
        self._statement_train: str = self._proto_train.format(epoch=self.epoch,
                                                              percent_complete=self.percent_complete,
                                                              samples_done=self._train_samples_done,
                                                              time_span=self._time_spent_train,
                                                              digits_epoch=self.digits_epoch,
                                                              digits_samples=self.digits_train_samples,
                                                              )
        return self._statement_train
    
    @train_statement.setter
    def train_statement(self, value: Dict[str, Union[int, float]]):
        """
        The setter updates the instance variables, if the corresponding member variables are present in
        the `value` dictionary. If a certain variable is not found, the corresponding instance variables
        are kept unchanged.
        Args:
            value:
        Returns:
        """
        # accepts fields: [step_count', 'samples_done'] and maps them to the train statement variables
        now = dt.now()
        self._step_count = value.get('step_count', self._step_count)
        self._train_samples_done = value.get('samples_done', self._train_samples_done)
        self._time_spent_train = self.get_epoch_time(now)
    
    @property
    def eval_statement(self) -> str:
        """  Updates the underlying statement, reflecting the current state of the
         instance variables, and returns the statement as formatted string"""
        self._statement_eval: str = self._proto_eval.format(samples_done=self._eval_samples_done,
                                                            digits_samples=self.digits_eval_samples,
                                                            time_span=self._time_spent_eval)
        return self._statement_eval
    
    @eval_statement.setter
    def eval_statement(self, value: Dict[str, Union[int, float]]):
        """
        The setter updates the instance variables, if the corresponding member variables are present in
        the `value` dictionary. If a certain variable is not found, the corresponding instance variables
        are kept unchanged.
        Args:
            value:
        Returns:
        """
        # accepts fields: [step_count', 'samples_done'] and maps them to the eval statement variables.
        now = dt.now()
        self._step_count = value.get('step_count', self._step_count)
        self._eval_samples_done = value.get('samples_done', self._eval_samples_done)
        self._time_spent_eval = value.get('time_span', self.get_eval_time(now))
    
    @property
    def ckpt_statement(self):
        return self._statement_ckpt
    
    @ckpt_statement.setter
    def ckpt_statement(self, value: Dict[str, bool]):
        self._checkpoint = 'checkpoint!' if \
            value.get('is_ckpt', False) else ''
        self._statement_ckpt = self._proto_ckpt.format(ckpt=self._checkpoint)
    
    def reset_statements(self):
        self.train_metrics = {}
        self.eval_metrics = {}
        self.train_statement = {'epoch': np.nan, 'samples_done': 0}
        self.eval_statement = {'samples_done': 0}
        self.ckpt_statement = {'is_ckpt': False}
    
    @property
    def terminal_size(self):
        try:
            terminal_size = os.get_terminal_size().columns - 1
        except OSError:
            terminal_size = 256
        return terminal_size
    
    def __repr__(self):
        if self.mode == ModeProgressBar.DEFAULT:
            statements = [re.sub(self._nan_pattern, '?? ', str(self.train_statement))]
            statements.append(self.train_metrics) if self.train_metrics else None
            if self.eval_metrics:
                statements.append(re.sub(self._nan_pattern, '?? ', str(self.eval_statement)))
                statements.append(self.eval_metrics)
            statements.append(self.ckpt_statement)
            statement = ' || '.join(statements)
        
        elif self.mode == ModeProgressBar.EVAL_ONLY:
            statements = [re.sub(self._nan_pattern, '?? ', str(self.eval_statement)),
                          self.eval_metrics]
            statement = ' || '.join(statements)
        
        else:
            raise ValueError
        return statement
    
    def __str__(self):
        statement = repr(self)
        statement = self.collapsed_statement(statement, self.terminal_size) \
            if self.terminal_size < len(statement) else statement
        return statement
    
    @staticmethod
    def collapsed_statement(statement, prescribed_len, sep=' || ', filler=' ... '):
        splits = statement.split(sep)
        splits.pop(-1) if not splits[-1] else None
        splits_to_print = [splits.pop(0)]
        
        def get_budget(_target, _splits):
            return prescribed_len - len(sep.join(splits_to_print))
        
        def collapsed_segment(segment_statement, budget, segment_sep=' | '):
            segments_to_print = [filler]
            segments = segment_statement.split(segment_sep)
            budget = budget - 2 * len(sep) - len(filler)
            for segment in segments[::-1]:
                seg_char_len = len(sep.join(segments_to_print)) + \
                               len(segment_sep) + len(segment)
                if seg_char_len < budget:
                    segments_to_print.insert(1, segment)
                else:
                    segments_to_print = [''] if len(segments_to_print) <= 1 \
                        else segments_to_print
                    break
            # if len(segment_sep.join(segments_to_print)) > budget:
            #     segments_to_print = ['']
            return segment_sep.join(segments_to_print)
        
        for k, split in enumerate(splits[::-1]):
            statement_char_len = len(sep.join(splits_to_print)) + \
                                 2 * len(sep) + len(filler) + len(split)
            if statement_char_len < prescribed_len:
                splits_to_print.insert(1, split)
            else:
                segment_budget = get_budget(prescribed_len, splits_to_print)
                seg_collapsed = collapsed_segment(split, segment_budget)
                splits_to_print.insert(1, seg_collapsed) if seg_collapsed else None
                budget_rest = get_budget(prescribed_len, splits_to_print)
                budget_rest = budget_rest - (len(sep))
                budget_rest = max(0, budget_rest)
                if k == len(splits) - 1:
                    splits_to_print.insert(1, f'{filler:^{budget_rest}}') \
                        if not seg_collapsed else splits_to_print[1].rjust(
                        len(splits_to_print[1]) + budget_rest)
                else:
                    splits_to_print.insert(1, f'{filler:^{budget_rest}}')
                break
        return sep.join(splits_to_print)


if __name__ == '__main__':
    # 1st statement
    prog_bar = ProgressBar()
    prog_bar.epoch = 1
    prog_bar.train_statement = {'samples_done': 0}
    prog_bar.train_metrics = {'acc/train': 2.7, 'loss/train': 3.9}
    prog_bar.eval_statement = {'samples_done': 200}
    prog_bar.eval_metrics = {'acc/val': 1.7, 'loss/val': 6.9}
    prog_bar.ckpt_statement = {'is_ckpt': True}
    print(prog_bar)
    prog_bar.reset_statements()
    # 2nd statement
    prog_bar.epoch = np.nan
    prog_bar.train_statement = {'samples_done': 0}
    prog_bar.train_metrics = {'acc/train': 2.7, 'loss/train': 3.9}
    prog_bar.eval_metrics = {'acc/val': 1.7, 'loss/val': 6.9}
    print(prog_bar, end='\n\n')
    prog_bar.reset_statements()
    # 3rd statement
    prog_bar.mode = ModeProgressBar.EVAL_ONLY
    prog_bar.eval_statement = {'samples_done': 200}
    prog_bar.eval_metrics = {'acc/val': 1.7, 'loss/val': 6.9}
    print(prog_bar)