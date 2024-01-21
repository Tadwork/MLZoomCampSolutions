import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import KFold

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# SEED = 65
n_splits = 10

arguments = argparse.ArgumentParser()
arguments.add_argument(
    "-i",
    "--input_filename",
    help="The filename of the model to convert relative to local (e.g. model-checkpoints\model.hdf5)",
)
arguments.add_argument(
    "-o", "--output_filename", help="The filename of the converted model"
)


def validate(train_dir, labels_df, model):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    batch_size = 32
    fold_no = 1
    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(labels_df):
        # print(test)
        val_datagen = ImageDataGenerator()
        target_size = (150, 150)
        generator_args = dict(
            directory=train_dir,
            x_col="ID",
            y_col="Class",
            class_mode="categorical",
            batch_size=batch_size,
            target_size=target_size,
        )
        val_generator = val_datagen.flow_from_dataframe(
            # select from labels_df the rows that are in the test set
            labels_df.iloc[test],
            **generator_args,
        )

        scores = model.evaluate(val_generator, verbose=0)
        print(
            f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
        )
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    print("------------------------------------------------------------------------")
    print("Score per fold")
    for i in range(0, len(acc_per_fold)):
        print(
            "------------------------------------------------------------------------"
        )
        print(f"> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%")
    print("------------------------------------------------------------------------")
    print("Average scores for all folds:")
    print(f"> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})")
    print(f"> Loss: {np.mean(loss_per_fold)}")
    print("------------------------------------------------------------------------")


def convert(model, output_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(output_filename, "wb") as f_out:
        f_out.write(tflite_model)


if __name__ == "__main__":
    current_directory = os.getcwd()
    labels_df = pd.read_csv(current_directory + "/face-age-detection/train.csv")

    train_dir = current_directory + "/face-age-detection/Train/"
    args = arguments.parse_args()
    print(f"Loading {args.input_filename}")
    model = keras.models.load_model(args.input_filename)
    validate(train_dir, labels_df, model)
    print(f"Converting {args.input_filename} to {args.output_filename}")
    convert(model, args.output_filename)
    print("Done")
