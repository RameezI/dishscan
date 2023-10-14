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

import os
import argparse
import pandas as pd
import tensorflow as tf
from pathlib import Path
import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
from dishscan.utils import base_parser
from sklearn.metrics import average_precision_score
from dishscan.datasets.nutrition5k import INGREDIENTS
from dishscan import DataStream, datastreams

sns.set_style("white")


def extract_data_stream(data_stream: DataStream, destination: os.PathLike):
    """
    Extract the given stream (images) on to the disk
    """

    os.makedirs(destination) if not os.path.isdir(destination) else None

    labels = []
    for k, batch in enumerate(data_stream):
        print(f"\rProcessing batch: {k + 1}/{len(data_stream)} ... ", end="")
        for _id, image, ingredients in zip(
            batch["id"], batch["image"], batch["ingredients"]
        ):
            file_id = Path(destination, _id.numpy().decode("utf-8"))
            tf.io.write_file(f"{file_id}.png", tf.io.encode_png(image))
            labels.append({"file_id": file_id.stem, "ingredients": ingredients.numpy()})

    df_metadata = pd.DataFrame(data=labels).set_index("file_id")
    df_metadata = df_metadata.ingredients.apply(pd.Series)
    df_metadata.to_csv(Path(destination, "metadata.csv"))
    print("Done!")


def write_predictions_csv(
    data_stream: DataStream,
    export_dir: os.PathLike,
    destination: os.PathLike,
    split_name="eval",
):
    """
    Predicts using a exported model and write predictions, along with the ground-truth, to a csv file.
    """
    df_predictions = pd.DataFrame()
    df_labels = pd.DataFrame()
    ingredients = {key - 1: value for key, value in INGREDIENTS.items()}

    interpreter = tf.saved_model.load(export_dir)
    for k, batch in enumerate(data_stream):
        index = [_id.decode("utf-8") for _id in batch["id"].numpy()]

        labels = batch["ingredients"]
        df_labels_batch = pd.DataFrame(labels.numpy(), index=index)
        df_labels_batch.rename(columns=ingredients, inplace=True)

        predictions = tf.sigmoid(interpreter.predict_step_tf(batch))
        df_predictions_batch = pd.DataFrame(predictions.numpy(), index=index)
        df_predictions_batch.rename(columns=ingredients, inplace=True)

        df_predictions = (
            df_predictions_batch
            if df_predictions.empty
            else pd.concat([df_predictions, df_predictions_batch], axis=0)
        )

        df_labels = (
            df_labels_batch
            if df_labels.empty
            else pd.concat([df_labels, df_labels_batch], axis=0)
        )

        print(
            f"\rPerformed predictions over {k+1}/{len(data_stream)} batches ... ",
            end="",
        )

    df = pd.concat([df_labels, df_predictions], axis=0, keys=["labels", "predictions"])
    df = df.rename_axis(("strain", "id"))

    print("\nPrediction dataframe populated.")
    print("writing to csv...", end="")
    df.to_csv(Path(destination, f"predictions_{split_name}.csv"))
    print("Done!")


# Helper function used for visualization of the layout of subplots
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def mean_average_precision_score(meta_df: pd.DataFrame):
    predictions = meta_df.loc["predictions"].to_numpy()
    labels = meta_df.loc["labels"].to_numpy()
    mAP = average_precision_score(labels, predictions, average="micro")
    print(f"calculating mAP over {len(predictions)} samples:")
    return mAP


def annotate_images(meta_df: pd.DataFrame, images_dir: Path, output_dir: Path):
    predictions_df = meta_df.loc["predictions"]
    labels_df = meta_df.loc["labels"]

    for k, (index, row) in enumerate(predictions_df.iterrows()):
        axes = plt.figure(constrained_layout=True, figsize=(12.8, 9.6)).subplot_mosaic(
            """
            AA.
            AAB
            AA.
            """,
            gridspec_kw={
                # set the height ratios between the rows
                "height_ratios": [1, 3.5, 1],
            },
        )
        axes["A"].set_axis_off()
        axes["B"].set_frame_on(False)
        axes["B"].set_autoscaley_on(True)

        output_dir.mkdir(parents=False, exist_ok=True)
        image = PIL.Image.open(Path(images_dir, f"{index}.png"))
        ingredients = row.sort_values(ascending=False).loc[row >= 0.6]

        # Draw image
        axes["A"].imshow(image)

        # Plot ingredient bars
        bar_colors = [
            "green" if labels_df.loc[index, ingredient] else "red"
            for ingredient in ingredients.index.values
        ]
        bars = axes["B"].barh(
            ingredients.index.values, ingredients.values, color=bar_colors
        )
        axes["B"].bar_label(
            bars,
            label_type="center",
            fmt="%.2f",
            color="w",
            fontfamily="monospace",
            fontsize="medium",
            fontweight="bold",
        )

        axes["B"].set_yticks(ingredients.index.values)
        axes["B"].set_yticklabels(
            ingredients.index.values,
            fontsize="x-large",
            fontfamily="monospace",
            fontweight="bold",
        )
        axes["B"].set_xticklabels([])
        axes["B"].set_ylim((-2, 15))
        axes["B"].invert_yaxis()  # labels read top-to-bottom

        print(
            f"\rprogress --> {k + 1}/{len(predictions_df)} records processed ... ",
            end="",
        )
        plt.savefig(Path(output_dir, f"{index}.png", dpi=1200))
        plt.close()
    print("done!")


def configuration() -> argparse.Namespace:
    """Returns the command line configuration"""
    parser = argparse.ArgumentParser(parents=[base_parser(training_component=False)])
    parser.set_defaults(logs_dir=str(Path(Path.home(), "tensorflow_logs")))
    parser.add_argument(
        "analysis_dir",
        type=str,
        required=False,
        default=str(Path(Path.home(), "dishscan_analysis")),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        default=False,
        help="Weather to re-extract the images to be analyzed from the data-stream.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = configuration()
    run_dir = Path(args.logs_dir, args.name, args.run_id)
    _images_dir = Path(args.analysis_dir, "images_to_eval")
    _output_dir = Path(args.analysis_dir, "annotated_images")

    _, eval_ds = datastreams(
        "nutrition5k",
        "/res448:1.*.*",
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        skip_normalization=args.skip_normalization,
    )
    # 1 write predictions for all data in the eval split
    write_predictions_csv(run_dir, destination=args.analysis_dir)
    df = pd.read_csv(Path(run_dir, "predictions_eval.csv"), index_col=["strain", "id"])
    print(f"Evaluating Model: {run_dir}")
    print(f"mAP: {mean_average_precision_score(df)}")

    # 2 generate images from DataStream [eval-split]
    if args.regenerate or not _images_dir.exists():
        # extracts the eval stream to disk
        extract_data_stream(eval_ds, _images_dir)

    # 3 generate annotated images
    annotate_images(df, _images_dir, _output_dir)