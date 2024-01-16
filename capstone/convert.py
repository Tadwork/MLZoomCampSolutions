import argparse

import tensorflow as tf
from tensorflow import keras

arguments = argparse.ArgumentParser()
arguments.add_argument("-i", "--input_filename", help="The filename of the model to convert")
arguments.add_argument("-o", "--output_filename", help="The filename of the converted model")

def convert(input_filename, output_filename):
    model = keras.models.load_model(input_filename)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(output_filename, 'wb') as f_out:
        f_out.write(tflite_model)

if __name__ == "__main__":
    args = arguments.parse_args()
    print(f"Converting {args.input_filename} to {args.output_filename}")
    convert(args.input_filename, args.output_filename)
    print("Done")