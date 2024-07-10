import os.path

import luigi
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import utils, ProjectConfig
from . import Consolidation


class Cleaning(luigi.Task):
    """
    Data cleaning by applying three different filters, based on timeframe (from 20:00 to 12:00) and on HRR-TC
    (Heart Rate Recovery Threshold Curve).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().cleaning_path
        self.current_patient = None
        self.model = self._create_model()
        # print("Coefficients of the model: ", self.model)

    def requires(self):
        return Consolidation()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        prev_files = utils.get_prev_files_path(self.input().path)

        path_list = []
        for prev_file in prev_files:
            self.current_patient = prev_file.split("/")[-1].split(".")[0].split("_")[
                1]  # {path}/{prev_file}_SX.csv -> SX
            patient_path = self._process_file(prev_file)
            path_list.append(patient_path)
        utils.create_output_paths_file(self.results_path, path_list)

    def _process_file(self, prev_file):
        print(f"Patient: {self.current_patient}")
        # date;time;accx;accy;accz;gyrosx;gyrosy;gyrosz;hr;stage
        original_file = pd.read_csv(prev_file, sep=";")
        print(f"\tOriginal file length: {len(original_file) - 1}")
        preliminary_file = self._preliminary_filter(original_file)
        time_windows = utils.create_time_windows(data=preliminary_file,
                                                 w_size=ProjectConfig().w_size,
                                                 w_overlapping=ProjectConfig().w_overlapping)
        final_file = self._create_final_file(time_windows)
        print(f"\tFinal file length: {len(final_file)}")
        lines_removed = len(original_file) - len(final_file) - 1
        print(f"\t - A total of {lines_removed} lines were removed")
        print(f"\tPercentage of data removed: {(lines_removed / (len(original_file) - 1)) * 100}%")
        path = os.path.join(self.results_path, f"cleaned_{self.current_patient}.csv")
        final_file.to_csv(path, index=False, sep=";")
        return path

    def _create_final_file(self, time_windows):
        print(f"\tFiltering by HRR using the following model coefficients: {self.model}...")
        final_data = []
        removed_windows = 0
        for window in tqdm(time_windows):
            if self._hrr_filter(window):
                csv_data = window.values.tolist()
                final_data.extend(csv_data)
            else:
                removed_windows += 1

        final_columns = ["date", "time", "accx", "accy", "accz", "gyrosx", "gyrosy", "gyrosz", "hr", "stage"]
        final_df = pd.DataFrame(final_data, columns=final_columns)
        final_df.drop_duplicates(inplace=True)
        print(f"\t - {removed_windows} windows were removed")
        return final_df

    def _preliminary_filter(self, file):
        print(f"\tFiltering by time [20:00 - 12:00]...")
        preliminary_file = file[file.apply(lambda x: self._time_filter(x["date"]), axis=1)]
        removed_lines = len(file) - len(preliminary_file)
        print(f"\t - {removed_lines} lines were removed")
        return preliminary_file

    def _hrr_filter(self, window_data):
        hr_list = window_data["hr"].astype(float).tolist()
        min_hr = min(hr_list)
        max_hr = max(hr_list)
        min_hr_list, max_hr_list = self._find_indexes(hr_list, min_hr, max_hr)
        # utils.plot_hr_variation(hr_list)

        time_diff1 = self._calculate_time_diff(max(min_hr_list), min(max_hr_list))
        time_diff2 = self._calculate_time_diff(min(min_hr_list), max(max_hr_list))

        time_diff = min(time_diff1, time_diff2)

        hr_diff = abs(min_hr - max_hr)
        hrr_threshold = self.model[0] + self.model[1] * np.log(time_diff)
        return 0 < hr_diff < hrr_threshold

    @staticmethod
    def _time_filter(data):
        data_hour = utils.str_to_date(data).hour
        from_hour = 20
        to_hour = 12
        return from_hour <= data_hour or data_hour < to_hour

    @staticmethod
    def _calculate_time_diff(min_hr_idx, max_hr_idx):
        return abs(min_hr_idx - max_hr_idx) / 10

    @staticmethod
    def _find_indexes(hr_list, min_hr, max_hr):
        min_hr_list, max_hr_list = [], []
        for hr_index, hr in enumerate(hr_list):
            if hr == min_hr:
                min_hr_list.append(hr_index)
            elif hr == max_hr:
                max_hr_list.append(hr_index)
        if min_hr == max_hr:
            max_hr_list = min_hr_list
        return min_hr_list, max_hr_list

    @staticmethod
    def _create_model():
        data = pd.DataFrame({
            'secs': [10, 20, 30, 40, 50],
            'ppm': [18.4 + 7.7, 24.2 + 8.9, 28.6 + 9.6, 31.9 + 10.1, 34.3 + 10.4]
        })
        model = np.polyfit(np.log(data['secs']), data['ppm'], 1)
        # utils.plot_model(data, model)
        return model
