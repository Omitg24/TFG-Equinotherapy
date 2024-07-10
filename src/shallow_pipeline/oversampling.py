import os
from typing import Dict, List

import luigi
import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

from . import ShallowPartitioning
from ..utils import ProjectConfig, utils


class ShallowOversampling(luigi.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().shallow_oversampling_path
        self.fold_id = "fold"
        self.leave_one_participant_id = "leave_one_participant"

    def requires(self):
        return ShallowPartitioning()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        partitions_files = utils.get_prev_files_path(self.input().path)
        partitions = utils.get_partitions(partitions_files)

        path_list = []
        path_list.extend(self._oversample(partitions, self.fold_id))
        path_list.extend(self._oversample(partitions, self.leave_one_participant_id))
        utils.create_output_paths_file(self.results_path, path_list)

    def _oversample(self, partitions: Dict, partition_type: str) -> List[str]:
        partitions = partitions[partition_type]
        sm = SMOTE(random_state=0, sampling_strategy='not majority')
        path_list = []
        for partition_name, data in tqdm(partitions.items()):
            path = os.path.join(self.results_path, partition_type)
            if "train" in partition_name:
                X = data.drop(columns=['stage'])
                y = data['stage']

                X_resampled, y_resampled = sm.fit_resample(X, y)

                resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

                path = os.path.join(path, f"{partition_name}.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                resampled_data.to_csv(path, index=False, sep=";")
            else:
                path = os.path.join(path, f"{partition_name}.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                data.to_csv(path, index=False, sep=";")
            path_list.append(path)
        return path_list
