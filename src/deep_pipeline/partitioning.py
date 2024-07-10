import os.path

import luigi
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from tqdm import tqdm

from ..preprocessing_pipeline import Scaling
from ..utils import ProjectConfig, utils


class DeepPartitioning(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().deep_partitions_path

    def requires(self):
        return Scaling()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        prev_files = utils.get_prev_files_path(self.input().path)
        all_data = pd.DataFrame()

        for prev_file in prev_files:
            current_patient = prev_file.split("/")[-1].split(".")[0].split("_")[
                1]
            patient_data = pd.read_csv(prev_file, sep=";")
            patient_data['patient_id'] = current_patient
            all_data = pd.concat([all_data, patient_data])

        path_list = self._process_data(all_data)
        utils.create_output_paths_file(self.results_path, path_list)

    def _process_data(self, all_data):
        all_data = utils.unify_sleep_stages(all_data, n_phases=ProjectConfig().n_phases)
        x = all_data[['accx', 'accy', 'accz', 'hr']]
        y = all_data['stage']
        groups = all_data['patient_id']

        print("Creating partitions...")
        path_list = self._create_leave_one_group_partitions(x, y, groups)
        return path_list

    def _create_leave_one_group_partitions(self, x, y, groups):
        leave_one_group_out = LeaveOneGroupOut()
        path_list = []
        for i, (train_index, test_index) in enumerate(leave_one_group_out.split(x, y, groups)):
            print(f"\tCreating partitions for participant {i}...")
            path = os.path.join(self.results_path, "patient_" + str(i) + "/")
            os.makedirs(path, exist_ok=True)

            X_train, X_validation = x.iloc[train_index], x.iloc[test_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[test_index]

            test_data = pd.concat([X_validation, y_validation], axis=1)
            test_file = os.path.join(path, f"test_participant_{i}.csv")
            test_data.to_csv(test_file, index=False, sep=";")

            path_list.append(test_file)
            path_list.extend(self._create_cross_validation_training_partitions(X_train, y_train, i, path))
            print(f"\t- Saved in {path}")
        return path_list

    @staticmethod
    def _create_cross_validation_training_partitions(x_train, y_train,
                                                     participant_index, path):
        skf = StratifiedKFold(n_splits=10)
        path_list = []
        for fold, (train_fold_index, val_fold_index) in tqdm(enumerate(skf.split(x_train, y_train))):
            X_train_fold, X_val_fold = x_train.iloc[train_fold_index], x_train.iloc[val_fold_index]
            y_train_fold, y_val_fold = y_train.iloc[train_fold_index], y_train.iloc[val_fold_index]

            fold_train_data = pd.concat([X_train_fold, y_train_fold], axis=1)
            fold_val_data = pd.concat([X_val_fold, y_val_fold], axis=1)

            fold_train_file = os.path.join(path, f"train_fold_{participant_index}_{fold}.csv")
            fold_val_file = os.path.join(path, f"validation_fold_{participant_index}_{fold}.csv")
            fold_train_data.to_csv(fold_train_file, index=False, sep=";")
            fold_val_data.to_csv(fold_val_file, index=False, sep=";")

            path_list.append(fold_train_file)
            path_list.append(fold_val_file)
        return path_list
