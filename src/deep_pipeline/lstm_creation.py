from keras import Sequential, Input
import keras.src.layers as layers
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.optimizers import Adam

from ..utils import ProjectConfig


# Obtained from https://github.com/shashankpr/sleep-classification/blob/master/src/lstm.py
def create_lstm1(number_inputs, window_size) -> Sequential:
    model = Sequential()

    model.add(Input(shape=(window_size, number_inputs)))

    model.add(layers.LSTM(units=50, return_sequences=True))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(units=100, return_sequences=False))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=ProjectConfig().n_phases, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    return model


# Obtained from
# https://github.com/dahengk/sleep_stage_LSTM/blob/main/lstm.py
def create_lstm2(number_inputs, window_size):
    model = Sequential()

    model.add(Input(shape=(window_size, number_inputs)))

    model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.1))

    model.add(layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.1))

    model.add(layers.LSTM(32, activation='relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(units=ProjectConfig().n_phases, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Obtained from
# https://github.com/Newber0/Automatic-Sleep-Stage-Classification-using-EEG-Data
# This is an inspiration
def create_lstm3(number_inputs, window_size):
    model = Sequential()

    model.add(Input(shape=(window_size, number_inputs)))

    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=ProjectConfig().n_phases, activation='softmax'))

    model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
    return model
