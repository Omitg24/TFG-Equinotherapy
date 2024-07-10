import math
from statistics import mode
from typing import Tuple, List

import keras
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import utils


class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 x_data: pd.DataFrame,
                 y_data: pd.DataFrame,
                 name: str = "DataGenerator",
                 window_size: int = 30,
                 window_overlap: int = 15,
                 batch_size: int = 32,
                 lstm_mode: int = 1,
                 n_clases: int = 5,
                 sample_frequency: int = 10,
                 shuffle: bool = True,
                 is_training: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.name = name
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.lstm_mode = lstm_mode
        self.sample_frequency = sample_frequency
        self.shuffle = shuffle
        self.is_training = is_training

        self.x_data, self.y_data = self._oversample()
        self.x_data, self.y_data = self._create_windows()
        encoder = LabelEncoder()
        self.y_data = encoder.fit_transform(self.y_data)
        self.y_data = keras.utils.to_categorical(self.y_data, num_classes=n_clases)
        self.indexes = self.on_epoch_end()

    def __len__(self) -> int:
        n = len(self.indexes)
        return math.ceil(n / self.batch_size) if n > 0 else 0

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        windows_x = []
        windows_y = []

        min_val = max(0, min(len(self.indexes), index * self.batch_size))
        max_val = min(len(self.indexes), (index + 1) * self.batch_size)
        for val in range(min_val, max_val):
            x_data = self.x_data[self.indexes[val]].copy()
            y_data = self.y_data[self.indexes[val]]
            x_data = self._scale_data(x_data)
            x_data = self._update_mode(x_data)

            windows_x.append(x_data)
            windows_y.append(y_data)

        return np.array(windows_x), np.array(windows_y)

    def _create_windows(self) -> Tuple[List, List]:
        windows_x = []
        windows_y = []
        n_windows = math.ceil(len(self.x_data) / (self.window_overlap * self.sample_frequency))
        for w in range(n_windows):
            start_index = w * self.window_overlap * self.sample_frequency
            end_index = min(len(self.x_data), start_index + (self.window_size * self.sample_frequency))
            indexes = list(range(start_index, end_index))
            if len(indexes) == (self.window_size * self.sample_frequency):    # All windows must be equal
                X = self.x_data.iloc[indexes]
                Y = mode(self.y_data[indexes])

                windows_x.append(X)
                windows_y.append(Y)

        return windows_x, windows_y

    def _oversample(self):
        if self.is_training:
            sm = SMOTE(random_state=0, sampling_strategy='not majority')
            X_resampled, y_resampled = sm.fit_resample(self.x_data, self.y_data)
            return X_resampled, y_resampled
        return self.x_data, self.y_data

    def _update_mode(self, x_data: pd.DataFrame):
        if self.lstm_mode == 2:
            magacc = utils.calculate_magnitude(x_data['accx'], x_data['accy'], x_data['accz'])
            ret = x_data.copy()
            ret['magacc'] = magacc
        elif self.lstm_mode == 3:
            magacc = utils.calculate_magnitude(x_data['accx'], x_data['accy'], x_data['accz'])
            ret = x_data[['hr']].copy()
            ret['magacc'] = magacc
        else:
            ret = x_data.copy()
        return ret

    def on_epoch_end(self):
        indexes = np.arange(len(self.x_data))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    @staticmethod
    def _scale_data(x_data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(x_data)
        return pd.DataFrame(scaled_data, columns=x_data.columns)


if __name__ == "__main__":
    train_fold = pd.read_csv("../../files/pipeline/deep_learning/partitions_data/patient_0/train_fold_0_0.csv", sep=";")
    validation_fold = pd.read_csv(
        "../../files/pipeline/deep_learning/partitions_data/patient_0/validation_fold_0_0.csv", sep=";")

    train_generator = DataGenerator(x_data=train_fold[['accx', 'accy', 'accz', 'hr']],
                                    y_data=train_fold['stage'],
                                    name="train",
                                    window_size=30,
                                    window_overlap=15,
                                    lstm_mode=2,
                                    shuffle=True)
    validation_generator = DataGenerator(x_data=validation_fold[['accx', 'accy', 'accz', 'hr']],
                                         y_data=validation_fold['stage'],
                                         name="validation",
                                         window_size=30,
                                         window_overlap=15,
                                         lstm_mode=2,
                                         is_training=False,
                                         shuffle=True)
    for i in range(train_generator.__len__()):
        a, b = train_generator.__getitem__(i)
        print(a)
        print(b)

    for i in range(validation_generator.__len__()):
        a, b = validation_generator.__getitem__(i)
        print(a)
        print(b)
