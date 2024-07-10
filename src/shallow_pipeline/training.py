import copy
import itertools
import os
import time
from multiprocessing import Manager

import joblib
import luigi
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from . import ShallowOversampling
from ..utils import ProjectConfig, utils
from .models import *


class ShallowTraining(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().shallow_training_path
        self.fold_id = "fold"
        self.leave_one_participant_id = "leave_one_participant"
        self.models_info = Manager().list()

    def requires(self):
        return ShallowOversampling()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "models_info.csv"))

    def run(self):
        partitions_files = utils.get_prev_files_path(self.input().path)
        partitions = utils.get_partitions(partitions_files)

        self._train_leave_one_out(partitions[self.leave_one_participant_id])
        self._train_stratified_kfold(partitions[self.fold_id])
        self._save_model_info()

    def _train_stratified_kfold(self, partitions):
        print("Stratified KFold")
        self._train_model(partitions, self.fold_id)

    def _train_leave_one_out(self, partitions):
        print("Leave One Participant Out")
        self._train_model(partitions, self.leave_one_participant_id)

    def _train_model(self, partitions, partition_type):
        result_path = os.path.join(self.results_path, partition_type)
        for model_name, model in shallow_models.items():
            start_time = time.time()
            if model_name not in param_grid.keys():
                print(f"Parameter grid not defined for {model_name}. Skipping training.")
                continue

            path = os.path.join(result_path, model_name)
            os.makedirs(path, exist_ok=True)

            print(f"\tTraining {model_name}")
            local_model_infos = Parallel(n_jobs=-1)(
                delayed(self.train_partition)(
                    partition_type,
                    i,
                    model_name,
                    model,
                    partitions,
                    path
                ) for i in range(len(partitions) // 2)
            )

            for local_model_info in local_model_infos:
                self.models_info.append(local_model_info)

            end_time = time.time()
            print(f"\t\tModels saved in {path}")
            print(f"\t\tTotal training time: {end_time - start_time} seconds.")
            self._save_model_info()

    @staticmethod
    def train_partition(partition_type, i, model_name, model, partitions, path):
        print(f"\t\tTraining {partition_type}_{i}")
        train_data = partitions[f'train_{partition_type}_{i}']
        validation_data = partitions[f'validation_{partition_type}_{i}']

        label_encoder = LabelEncoder()
        label_encoder.fit(train_data['stage'])
        X_train = train_data.drop(columns=['stage'])
        y_train = label_encoder.transform(train_data['stage'])
        X_validation = validation_data.drop(columns=['stage'])
        y_validation = label_encoder.transform(validation_data['stage'])

        params = param_grid[model_name]
        param_values = list(params.values())
        param_combinations = list(itertools.product(*param_values))

        best_f1_score = -1
        best_model_info = None

        for idx, combination in enumerate(param_combinations):
            param_combination = dict(zip(params.keys(), combination))
            cloned_model = copy.deepcopy(model)
            cloned_model.set_params(**param_combination)

            cloned_model.fit(X_train, y_train)
            predictions = cloned_model.predict(X_validation)
            accuracy = accuracy_score(y_validation, predictions)
            f1 = f1_score(y_validation, predictions, average='weighted')
            # TODO: ROC CURVE CANNOT BE USED BECAUSE ONE PATIENT DOEST NOT HAVE REM PHASE
            # roc_auc = roc_auc_score(y_validation, cloned_model.predict_proba(X_validation), multi_class='ovr',
            #                         average='weighted')

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_info = {
                    "model": cloned_model,
                    "model_name": model_name,
                    "validation_method": partition_type,
                    "partition": i,
                    "param_idx": idx,
                    "param_combinations": param_combination,
                    "accuracy": accuracy,
                    "f1_score": f1
                    # "roc_auc": roc_auc
                }

        if best_model_info is not None:
            model_filename = f"{model_name}_best_{partition_type}_{i}_P{best_model_info.get('param_idx')}.pkl"
            model_path = os.path.join(path, model_filename)
            joblib.dump(best_model_info["model"], model_path)
            best_model_info["model_path"] = model_path

        return best_model_info

    def _save_model_info(self):
        filename = os.path.join(self.results_path, "models_info.csv")
        with open(filename, 'w') as file:
            file.write(
                "model;validation_method;partition;param_idx;param_combinations;"
                "accuracy;f1_score;model_path\n")
            for info in self.models_info:
                model_name = info['model_name']
                validation_method = info['validation_method']
                partition = info['partition']
                param_idx = info['param_idx']
                param_combinations = "{" + ','.join(
                    [f"{key}={value}" for key, value in info['param_combinations'].items()]) + "}"
                accuracy = info['accuracy']
                f1 = info['f1_score']
                # roc_auc = info['roc_auc']
                model_path = info['model_path']
                file.write(
                    f"{model_name};{validation_method};{partition};{param_idx};{param_combinations};"
                    f"{accuracy};{f1};{model_path}\n")
        print(f"\tModels info saved in {filename}, containing a total of {len(self.models_info)} models.")
