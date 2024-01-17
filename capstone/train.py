"""
Training the final model
Saving it to a file (e.g. pickle)
"""
import os

from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess

SEED=42

def create_model(learning_rate, size_inner,droprate,  shape = (150,150,3)):
    inception = InceptionResNetV2(include_top=False, weights='imagenet')

    inception.trainable = False

    inputs = keras.Input(shape=shape)

    x = preprocess(inputs)
    x = inception(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dense(size_inner, activation='relu')(x)
    x = keras.layers.Dropout(droprate)(x)
    
    outputs = keras.layers.Dense(3, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # set to legacy optimizer to be compatible with Mac M1
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate
    )
    loss = keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_crossentropy",
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def train(train_dir, df_val, df_train, model,epochs=10):
    checkpoint = keras.callbacks.ModelCheckpoint(
        'model-checkpoints/inception_{epoch:02d}_{val_accuracy:.3f}.hdf5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
    )
    target_size = (150,150)

    batch_size = 32

    generator_args = dict(
        directory=train_dir,
        x_col='ID',
        y_col='Class',
        class_mode='categorical',
        batch_size=batch_size,
        target_size=target_size
    )
    # Training data generator
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        **generator_args
    )

    # Validation data generator
    val_datagen = ImageDataGenerator()

    val_generator = val_datagen.flow_from_dataframe(
        df_val,
        **generator_args
    )
    model.fit(train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )
    return model
    
def validate(train_dir,model, df_test, batch_size=32):
    val_datagen = ImageDataGenerator()
    target_size = (150,150)
    generator_args = dict(
        directory=train_dir,
        x_col='ID',
        y_col='Class',
        class_mode='categorical',
        batch_size=batch_size,
        target_size=target_size
    )
    val_generator = val_datagen.flow_from_dataframe(
        df_test,
        **generator_args
    )
    return model.evaluate(val_generator)

if __name__ == '__main__':
    current_directory = os.getcwd()
    labels_df = pd.read_csv(current_directory + '/face-age-detection/train.csv')
    df_train, df_test = train_test_split(labels_df, test_size=0.4, random_state=SEED)
    df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=SEED)

    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    
    train_dir = current_directory + '/face-age-detection/Train/'
    model = create_model(learning_rate=0.0001, size_inner=64,droprate=0.5)
    trained_model = train(train_dir, df_val, df_train, model,epochs=45)
    loss, accuracy = validate(train_dir,trained_model, df_test, batch_size=32)
    print('accuracy on test: ', accuracy)
    print('model weights saved to to the model-checkpoints directory')
