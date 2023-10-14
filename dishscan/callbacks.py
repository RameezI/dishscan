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
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict
from datetime import datetime as dt
from typing import Union, Collection
from .protocol import Protocol
from .progressbar import ProgressBar
from .utils import Number, VerbosityLevel

import logging

logger = logging.getLogger(__name__)

# Aliases
callbacks = tf.keras.callbacks


class Callback(callbacks.Callback):
    def __init__(self):
        super(Callback, self).__init__()
        self._run_dir: os.PathLike[str] = Path()
        self._protocol: Union[Protocol, None] = None
        self._writers: Union[None, Dict[str, tf.summary.SummaryWriter]] = None
        self._models: Dict[str, tf.keras.Model] = {}

    @property
    def models(self):
        """
        Returns: The models handled by the trainer; which are accessible to all callbacks
        """
        return self._models

    @models.setter
    def models(self, value: Dict[str, tf.keras.Model]):
        """
        Args:
            value: Set models accessible to all callback objects
        Returns:
        """
        self._models = value

    @property
    def writers(self) -> Union[None, Dict[str, tf.summary.SummaryWriter]]:
        """
        Returns a dictionary of summary writers
        """
        return self._writers

    @writers.setter
    def writers(self, value: Dict[str, tf.summary.SummaryWriter]):
        """
        Args:
            value:
        Returns:
        """
        self._writers = value

    @property
    def run_dir(self):
        return self._run_dir

    @run_dir.setter
    def run_dir(self, value: os.PathLike):
        self._run_dir = value

    @property
    def protocol(self):
        return self._protocol

    @protocol.setter
    def protocol(self, value: Protocol):
        self._protocol = value


class TensorBoardLogger(callbacks.Callback):
    def __init__(
        self,
        summaries_dir: Union[str, os.PathLike],
        frequency: int = 1,
        mode="batch",
        produce_graph: bool = True,
    ):
        super(TensorBoardLogger, self).__init__()
        if not isinstance(mode, str):
            raise TypeError

        if mode not in ["batch", "epoch"]:
            raise ValueError(
                f"Value Error, while setting mode!\n"
                f"The logger mode must be set to either `epoch` or `batch`\n"
                f"Found: {mode}"
            )

        self.summaries_dir = summaries_dir
        self.produce_graph = produce_graph
        self.frequency: Union[int, str] = frequency
        self.current_step: int = 0
        self.mode = mode

    def on_train_batch_end(self, batch, logs=None):
        writer = self.writers.get("train", tf.summary.create_noop_writer())
        if self.mode == "batch" and batch % self.frequency == 0:
            self.current_step = self.current_step + 1
            with writer.as_default(step=self.current_step):
                for name, value in filter(
                    lambda x: x[0].endswith("train"), logs.items()
                ):
                    tf.summary.scalar(name, value)

    def on_test_batch_end(self, batch, logs=None):
        writer = self.writers.get("test", tf.summary.create_noop_writer())
        if self.mode == "batch" and batch % self.frequency == 0:
            self.current_step = self.current_step + 1
            with writer.as_default(step=self.current_step):
                for name, value in filter(lambda x: x[0].endswith("val"), logs.items()):
                    tf.summary.scalar(name, value)

    def on_epoch_end(self, epoch, logs=None):
        train_writer = self.writers.get("train", tf.summary.create_noop_writer())
        val_writer = self.writers.get("test", tf.summary.create_noop_writer())
        if self.mode == "epoch" and epoch % self.frequency == 0:
            self.current_step = self.current_step + 1
            # Train writer
            with train_writer.as_default(step=self.current_step):
                for name, value in filter(
                    lambda x: x[0].endswith("train"), logs.items()
                ):
                    tf.summary.scalar(name, value)
            # Validation writer
            with val_writer.as_default(step=self.current_step):
                for name, value in filter(lambda x: x[0].endswith("val"), logs.items()):
                    tf.summary.scalar(name, value)


class CheckPointer(Callback):
    def __init__(
        self,
        checkpoint_dir: os.PathLike,
        monitor: str = "loss/val",
        silent: bool = False,
        keep_best: bool = True,
        mode: str = "auto",
        max_to_keep: int = 3,
        **kwargs,
    ):
        """
        Args:
            checkpoint_dir:         The directory where the checkpoints are created.
            monitor:                The monitored variable
            silent:                 Weather to print updates on checkpointing event
            keep_best:              Weather to keep the best, based on the monitored value, or the last checkpoint.
            mode:                   How to determine the best checkpoint based on the monitored variable,
                                    accepts one of [`min`, `max`, `auto`]
            max_to_keep:            Maximum number of checkpoint to keep.
            progress_bar:           Optionally accepts a progress bar used to update the check-point statement.

        """

        super(CheckPointer, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.silent = silent
        self.keep_best = keep_best
        self.monitor = monitor
        self.max_to_keep = max_to_keep
        self._current_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._progress_bar = kwargs.get("progress_bar", None)

        os.makedirs(str(self.checkpoint_dir)) if not os.path.exists(
            self.checkpoint_dir
        ) else None

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                f"ModelCheckpoint mode {mode} is unknown, fallback to auto mode."
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if self.monitor.startswith(("acc", "mAP", "fmeasure")):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        # Lazily create, at the start of the training, a checkpoint manager
        self.ckpt_manager = None

    def on_train_begin(self, logs: Union[None, Dict[str, Number]] = None):
        checkpoint = tf.train.Checkpoint(
            epoch=self._current_epoch,
            **self.models,
            objectives=self._protocol.objectives,
        )

        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            self.checkpoint_dir,
            checkpoint_name="ckpt",
            max_to_keep=self.max_to_keep,
        )
        if self.keep_best and logs:
            initial_best = logs.get(self.monitor)
            if initial_best is None:
                print("", file=sys.stderr)
                logger.warning(
                    f"\n{'Missing monitored-key':-^35}\n"
                    f"The checkpointer callback did not find the monitored key in the initial logs,"
                    f"\nif it is not found during training; will keep the {self.max_to_keep} most"
                    f" recent points instead!\n"
                )

        self.best = logs.get(self.monitor, self.best)

    def on_epoch_end(self, epoch, logs: Union[None, Dict[str, Number]] = None):

        self._current_epoch.assign(epoch)

        if self.monitor not in logs:
            self.keep_best = False

        if self.keep_best:
            current = logs.get(self.monitor, None)
            if self.monitor_op(current, self.best):
                self.ckpt_manager.save()
                if self._progress_bar is not None:
                    self._progress_bar.ckpt_statement = {"is_ckpt": True}
                else:
                    print(
                        f"checkpoint: {self.best}-->{current}!"
                    ) if not self.silent else None
                self.best = current
        else:
            self.ckpt_manager.save()
            if self._progress_bar is not None:
                self._progress_bar.ckpt_statement = {"is_ckpt": True}
            else:
                print("checkpoint!") if not self.silent else None


class ProgressParaphraser(Callback):
    """Callback that prints (formatted) progress and metrics to stdout."""

    def __init__(
        self,
        progress_bar: ProgressBar,
        verbose: VerbosityLevel = VerbosityLevel.UPDATE_AT_BATCH,
    ) -> None:
        super(ProgressParaphraser, self).__init__()
        self._progress_bar = progress_bar
        self.eval_statement_only = False
        self.verbose = verbose

    def on_train_begin(self, logs: Union[None, Dict[str, Number]] = None):
        # get max epochs and set the progress bar accordingly
        max_epochs = logs.pop("max_epochs", np.nan)
        self._progress_bar.max_epochs = max_epochs
        print(f"Initiating Train/Validate Cycle: ", end="") if self.verbose not in [
            VerbosityLevel.KEEP_SILENT
        ] else None

    def on_epoch_begin(self, epoch, logs: Union[None, Dict[str, Number]] = None):
        # prepare statement
        self._progress_bar.reset_statements()
        self._progress_bar.epoch = epoch
        self._progress_bar.train_samples = logs.get("samples/train", np.nan)
        self._progress_bar.epoch_start_time = dt.now()
        print(end="\n") if self.verbose not in [VerbosityLevel.KEEP_SILENT] else None

    def on_test_begin(self, logs: Union[None, Dict[str, Number]] = None):
        # prepare statement
        self._progress_bar.eval_samples = logs.get("samples/eval", np.nan)
        self._progress_bar.eval_start_time = dt.now()

    def on_train_batch_end(
        self, batch_index, logs: Union[None, Dict[str, Number]] = None
    ):
        # update statement
        if batch_index == 0:
            self._progress_bar.time_after_first_batch = dt.now()
        step_count = batch_index + 1
        samples_done = logs.pop("samples_done", np.nan)
        self._progress_bar.train_metrics = logs
        self._progress_bar.train_statement = {
            "step_count": step_count,
            "samples_done": samples_done,
        }
        print(f"\r{self._progress_bar}", end="") if self.verbose in [
            VerbosityLevel.UPDATE_AT_BATCH
        ] else None

    def on_test_batch_end(
        self, batch_index, logs: Union[None, Dict[str, Number]] = None
    ):
        # update statement
        if batch_index == 0:
            self._progress_bar.time_after_first_batch = dt.now()
        step_count = batch_index + 1
        samples_done = logs.pop("samples_done", np.nan)
        self._progress_bar.eval_metrics = logs
        self._progress_bar.eval_statement = {
            "step_count": step_count,
            "samples_done": samples_done,
        }
        print(f"\r{self._progress_bar}", end="") if self.verbose in [
            VerbosityLevel.UPDATE_AT_BATCH
        ] else None

    def on_test_end(self, logs: Union[None, Dict[str, Number]] = None):
        print(f"\r{self._progress_bar}", end="") if self.verbose not in [
            VerbosityLevel.KEEP_SILENT
        ] else None

    def on_epoch_end(self, epoch, logs=None):
        self._progress_bar.epoch = epoch
        self._progress_bar.train_samples = logs.get("samples/train", np.nan)
        self._progress_bar.eval_samples = logs.get("samples/eval", np.nan)
        print(f"\r{self._progress_bar}", end="") if self.verbose not in [
            VerbosityLevel.KEEP_SILENT
        ] else None


class Exporter(Callback):
    def __init__(
        self, checkpoint_dir: os.PathLike, signature: Collection[tf.TensorSpec] = ()
    ):
        super(Exporter, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self._epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.int64)
        if signature:
            self.signature = (
                signature if isinstance(signature, (list, tuple)) else (signature,)
            )

    def _load_checkpoint(self):
        checkpoint = tf.train.Checkpoint(
            epoch=self._epoch,
            **self.protocol.models,
            objectives=self._protocol.objectives,
        )
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_ckpt is None:
            raise FileNotFoundError
        status = checkpoint.restore(latest_ckpt)
        # status.expect_partial()
        return status

    def on_train_end(self, _: Union[None, Dict[str, Number]] = None):
        try:
            ckpt_load_status = self._load_checkpoint()
            ckpt_load_status.assert_existing_objects_matched()
            export_dir = Path(self.run_dir, "export")
            export_dir.mkdir(exist_ok=True)

        except FileNotFoundError:
            sys.stdout.flush()
            logger.warning(
                "\ncheckpoint file not found by the checkpoint manager, skipping export!"
            )

        except AssertionError as e:
            logger.error(str(e))
            raise AssertionError(
                "Checkpoint load failure!\n"
                f"checkpoint directory: {self.checkpoint_dir}"
            )
        else:
            signature = getattr(self, 'signature', ())
            self.protocol.export(export_dir, signature)