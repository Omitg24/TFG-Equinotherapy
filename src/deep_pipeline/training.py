import gc
import multiprocessing
import os
import shutil
import tracemalloc

import keras
import luigi
import numpy as np
import pandas as pd

from . import DeepPartitioning
from .data_generator import DataGenerator
from .lstm_creation import *
from ..utils import ProjectConfig, utils


class DeepTraining(luigi.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().deep_training_path
        self.inputs_by_mode = {
            1: 4,  # accx, accy, accz, hr
            2: 5,  # accx, accy, accz, magnitude, hr
            3: 2  # magnitude, hr
        }

    def requires(self):
        return DeepPartitioning()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        prev_files = utils.get_prev_files_path(self.input().path)

        patient_partitions = self._get_partitions(prev_files)

        path_list = self._run_experiments(patient_partitions)
        utils.create_output_paths_file(self.results_path, path_list)

    def _run_experiments(self, partitions):
        path_list = []

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2, maxtasksperchild=1) as pool:
            results = pool.map(self._process_participant_wrapper, partitions.items())

        for result in results:
            path_list.extend(result)

        return path_list

    def _process_participant_wrapper(self, participant_partitions):
        participant, partitions = participant_partitions
        return self._process_participant(participant, partitions)

    def _process_participant(self, participant, participant_partitions):
        path_list = []
        os.makedirs(os.path.join(self.results_path, participant), exist_ok=True)
        patient_idx = participant.split("_")[1]
        train_folds, validation_folds, test = self._get_folds(participant_partitions, patient_idx)
        model = self._create_neural_network()
        print(f"\tStarting participant {patient_idx}")
        tracemalloc.start()

        for ep in range(ProjectConfig().n_epochs):
            epoch_path = os.path.join(self.results_path, participant, f"epoch_{ep}")
            os.makedirs(epoch_path, exist_ok=True)
            print(f"\t\t{participant}Starting epoch {ep}")
            history = []
            for k in range(ProjectConfig().n_splits):
                train_fold = pd.read_csv(train_folds[k], sep=";")
                validation_fold = pd.read_csv(validation_folds[k], sep=";")
                print(f"\t\t\tTraining model for participant {participant} - epoch {ep} - fold {k}")
                hist = self._train_fold(model, train_fold, validation_fold)
                history.append(hist)

                current, peak = tracemalloc.get_traced_memory()
                print(f"\t\t\t{participant}Memory usage after fold {k}: Current = {current / 10 ** 6} MB; Peak = {peak / 10 ** 6} MB")

                del train_fold
                del validation_fold
                gc.collect()
            final_history = {}
            for key in history[0].history.keys():
                final_history.update({key: np.concatenate([hist.history[key] for hist in history])})

            current, peak = tracemalloc.get_traced_memory()
            print(f"\t\t{participant}Memory usage after epoch {ep}: Current = {current / 10 ** 6} MB; Peak = {peak / 10 ** 6} MB")
            history_path = os.path.join(epoch_path, f"history_{participant}_{ep}_all_folds.csv")
            pd.DataFrame.from_dict(final_history).to_csv(history_path, sep=";")
            path_list.append(history_path)
            print(f"\t\t{participant}Finished epoch {ep}")
            print(f"\t\t{participant}Saving lstm ({ProjectConfig().neural_network}) weights")
            weights_path = os.path.join(epoch_path, f"weights_{patient_idx}_{ep}.weights.h5")
            model.save_weights(weights_path)
            model_path = os.path.join(epoch_path, f"model_{patient_idx}_{ep}.keras")
            model.save(model_path)
            del model
            keras.backend.clear_session()
            path_list.append(weights_path)
            model = keras.saving.load_model(model_path)
        print(f"\t{participant}Finished participant {patient_idx}")
        print(f"\t{participant}Saving lstm ({ProjectConfig().neural_network}) model")
        model_path = os.path.join(self.results_path, participant, f"model_{patient_idx}.keras")
        model.save(model_path)
        path_list.append(model_path)

        test_path = os.path.join(self.results_path, participant, f"test_{patient_idx}.csv")
        shutil.copyfile(test, test_path)
        path_list.append(test_path)

        current, peak = tracemalloc.get_traced_memory()
        print(f"\t{participant}Memory usage after all epochs: Current = {current / 10 ** 6} MB; Peak = {peak / 10 ** 6} MB")

        del model
        gc.collect()
        keras.backend.clear_session()
        current, peak = tracemalloc.get_traced_memory()
        print(f"\tMemory usage after patient {participant}: Current = {current / 10 ** 6} MB; Peak = {peak / 10 ** 6} MB")

        tracemalloc.stop()
        return path_list

    def _create_neural_network(self):
        mode = ProjectConfig().lstm_mode
        neural_network = ProjectConfig().neural_network
        number_inputs = self.inputs_by_mode[mode]
        window_size = (ProjectConfig().w_size * ProjectConfig().sample_frequency)

        # All neural networks are already compiled
        if neural_network == 1:
            print("Creating LSTM1 model")
            return create_lstm1(number_inputs, window_size)
        elif neural_network == 2:
            print("Creating LSTM2 model")
            return create_lstm2(number_inputs, window_size)
        elif neural_network == 3:
            print("Creating LSTM3 model")
            return create_lstm3(number_inputs, window_size)

    def _train_fold(self, model, train_fold, validation_fold):
        X_train, y_train = (train_fold[['accx', 'accy', 'accz', 'hr']], train_fold['stage'])
        X_validation, y_validation = (validation_fold[['accx', 'accy', 'accz', 'hr']], validation_fold['stage'])
        hist = self._train_model(model, X_train, y_train, X_validation, y_validation)
        return hist

    @staticmethod
    def _train_model(model, x_train, y_train, x_validation, y_validation):
        train_generator = DataGenerator(x_data=x_train,
                                        y_data=y_train,
                                        name="train",
                                        window_size=ProjectConfig().w_size,
                                        window_overlap=ProjectConfig().w_overlapping,
                                        lstm_mode=ProjectConfig().lstm_mode,
                                        sample_frequency=ProjectConfig().sample_frequency,
                                        n_clases=ProjectConfig().n_phases)
        validation_generator = DataGenerator(x_data=x_validation,
                                             y_data=y_validation,
                                             name="validation",
                                             window_size=ProjectConfig().w_size,
                                             window_overlap=ProjectConfig().w_overlapping,
                                             lstm_mode=ProjectConfig().lstm_mode,
                                             n_clases=ProjectConfig().n_phases,
                                             sample_frequency=ProjectConfig().sample_frequency,
                                             is_training=False)

        hist = model.fit(train_generator,
                         steps_per_epoch=train_generator.__len__(),
                         epochs=1,
                         validation_data=validation_generator,
                         validation_steps=validation_generator.__len__())

        return hist

    @staticmethod
    def _get_partitions(prev_files):
        partitions = {}
        for prev_file in prev_files:
            patient = os.path.basename(os.path.dirname(prev_file))
            file_name = os.path.basename(prev_file).split(".")[0]
            partitions.setdefault(patient, {}).setdefault(file_name, prev_file)
        return partitions

    @staticmethod
    def _get_folds(partitions, patient_idx):
        train_folds = []
        validation_folds = []
        for i in range(ProjectConfig().n_splits):
            train_folds.append(partitions[f"train_fold_{patient_idx}_{i}"])
            validation_folds.append(partitions[f"validation_fold_{patient_idx}_{i}"])
        test = partitions[f"test_participant_{patient_idx}"]
        return train_folds, validation_folds, test
