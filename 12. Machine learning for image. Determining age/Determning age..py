from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_train(path):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=90, width_shift_range=0.2,
                                       height_shift_range=0.2)

    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)




    return train_datagen_flow

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=6, padding='same', kernel_size=(5, 5), activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None))
    model.add(Conv2D(kernel_size=(5, 5), filters=12, padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2),  strides=None))
    model.add(Conv2D(kernel_size=(5, 5), filters=24, padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=13, steps_per_epoch=None, validation_steps=None):
    train_datagen_flow = train_data
    val_datagen_flow = test_data
    model.fit(train_datagen_flow, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps, validation_data=val_datagen_flow, verbose=2)
    return model

