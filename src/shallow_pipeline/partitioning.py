import os

import luigi
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from tqdm import tqdm

from ..preprocessing_pipeline import Preprocessing
from ..utils import utils, ProjectConfig


class ShallowPartitioning(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().shallow_partitions_path

    def requires(self):
        return Preprocessing()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        prev_files = utils.get_prev_files_path(self.input().path)
        all_data = pd.DataFrame()

        for prev_file in prev_files:
            current_patient = prev_file.split("/")[-1].split(".")[0].split("_")[
                1]  # {path}/{prev_file}_SX.csv -> SX
            patient_data = pd.read_csv(prev_file, sep=";")
            patient_data['patient_id'] = current_patient
            all_data = pd.concat([all_data, patient_data])

        path_list = self._process_data(all_data)
        utils.create_output_paths_file(self.results_path, path_list)

    def _process_data(self, all_data):
        all_data = utils.unify_sleep_stages(all_data, n_phases=ProjectConfig().n_phases)
        x = all_data.drop(columns=['stage', 'patient_id'])
        y = all_data['stage']
        groups = all_data['patient_id']

        test_percentage = ProjectConfig().test_percentage

        train_x, test_x, train_y, test_y = (train_test_split(x, y, test_size=test_percentage, random_state=0)
                                            if test_percentage > 0
                                            else (x, None, y, None))
        path_list = []
        path_list.extend(self._create_cross_validation_partitions(train_x, train_y))
        path_list.extend(self._create_leave_one_group_partitions(train_x, train_y, groups))

        if test_percentage > 0:
            path_list.append(self._save_test_set(test_x, test_y))
        return path_list

    def _create_cross_validation_partitions(self, x: pd.DataFrame, y: pd.DataFrame):
        path = os.path.join(self.results_path, "fold/")
        os.makedirs(path, exist_ok=True)
        n_splits = ProjectConfig().n_splits
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        print(f"\tCreating Cross-Validation partitions...")
        path_list = []
        for i, (train_index, test_index) in tqdm(enumerate(stratified_kfold.split(x, y))):
            X_train, X_validation = x.iloc[train_index], x.iloc[test_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[test_index]
            train_data = pd.concat([X_train, y_train], axis=1)
            validation_data = pd.concat([X_validation, y_validation], axis=1)
            train_file = os.path.join(path, f"train_fold_{i}.csv")
            validation_file = os.path.join(path, f"validation_fold_{i}.csv")
            train_data.to_csv(train_file, index=False, sep=";")
            validation_data.to_csv(validation_file, index=False, sep=";")
            path_list.append(train_file)
            path_list.append(validation_file)
        return path_list

    def _create_leave_one_group_partitions(self, x: pd.DataFrame, y: pd.DataFrame, groups: pd.DataFrame):
        path = os.path.join(self.results_path, "leave_one_participant/")
        os.makedirs(path, exist_ok=True)
        leave_one_group_out = LeaveOneGroupOut()
        print(f"\tCreating Leave-One-Participant partitions...")
        path_list = []
        for i, (train_index, test_index) in tqdm(enumerate(leave_one_group_out.split(x, y, groups))):
            X_train, X_validation = x.iloc[train_index], x.iloc[test_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[test_index]
            train_data = pd.concat([X_train, y_train], axis=1)
            validation_data = pd.concat([X_validation, y_validation], axis=1)
            train_file = os.path.join(path, f"train_leave_one_participant_{i}.csv")
            validation_file = os.path.join(path, f"validation_leave_one_participant_{i}.csv")
            train_data.to_csv(train_file, index=False, sep=";")
            validation_data.to_csv(validation_file, index=False, sep=";")
            path_list.append(train_file)
            path_list.append(validation_file)
        return path_list

    def _save_test_set(self, test_x, test_y):
        test_data = pd.concat([test_x, test_y], axis=1)
        test_file = os.path.join(self.results_path, "test", "test_set.csv")
        test_data.to_csv(test_file, index=False, sep=";")
        return test_file
