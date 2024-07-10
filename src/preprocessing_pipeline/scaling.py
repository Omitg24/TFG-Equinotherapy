import os
from typing import List, Union, Dict

import luigi
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from . import Cleaning
from ..utils import utils, ProjectConfig


class Scaling(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().scaling_path

    def requires(self):
        return Cleaning()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        prev_files = utils.get_prev_files_path(self.input().path)

        # data = [pd.read_csv(prev_file.path, sep=";") for prev_file in prev_files]
        # for i, patient_data in enumerate(data):
        #     patient_data["magacc"] = utils.calculate_magnitude(patient_data["accx"], patient_data["accy"],
        #                                                        patient_data["accz"])
        # utils.plot_boxplot(data, "magacc")
        # utils.plot_boxplot(data, "hr")

        adjusted_patients = self._adjust_patients(prev_files)
        path_list = self._scale_patients(adjusted_patients)
        utils.create_output_paths_file(self.results_path, path_list)

    def _adjust_patients(self, prev_files):
        print("\nAdjusting patients...")
        output_path = os.path.join(self.results_path, "median_adjust")
        os.makedirs(output_path, exist_ok=True)
        adjusted_patients = []
        for prev_file in tqdm(prev_files):
            patient_id = prev_file.split("/")[-1].split(".")[0].split("_")[1]
            data = pd.read_csv(prev_file, sep=";")
            for column in ["accx", "accy", "accz", "hr"]:
                baseline_column = f"{column}_baseline"
                data[column] = data[column] - data[column].mean()
                data[baseline_column] = self.polyfit_baseline(data[column].values, degree=3)
            utils.plot_baseline(data, ["accx", "accy", "accz", "hr"], patient_id)
            data.drop(columns=["accx_baseline", "accy_baseline", "accz_baseline", "hr_baseline"], inplace=True)
            data["patient_id"] = patient_id
            data.to_csv(os.path.join(output_path, f"adjusted_{patient_id}.csv"), sep=";", index=False)
            adjusted_patients.append(data)
        return adjusted_patients

    def _scale_patients(self, data_list):
        print("\nScaling patients...")
        scaled_patients = []
        all_data = pd.concat(data_list, axis=0)
        columns_to_scale = ["accx", "accy", "accz", "hr"]
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(all_data[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=all_data.index)
        all_data[columns_to_scale] = scaled_df
        descriptions = []
        path_list = []
        for data in tqdm(data_list):
            patient_id = data["patient_id"].iloc[0]
            patient_data = all_data[all_data["patient_id"] == patient_id]
            patient_data.insert(patient_data.columns.get_loc("gyrosx"), "magacc", 0.0)
            patient_data.loc[:, "magacc"] = utils.calculate_magnitude(patient_data["accx"], patient_data["accy"],
                                                                      patient_data["accz"])
            description = patient_data[columns_to_scale + ["magacc"]].describe()
            description["patient_id"] = patient_id
            descriptions.append(description)
            # ChartGeneration("robust_scaling").generate_plot(patient_data, patient_id, mag_max_limit=5,
            #                                                 hr_max_limit=10, mag_min_limit=-5, hr_min_limit=-3)
            scaled_patients.append(patient_data)
            path = os.path.join(self.results_path, f"scaled_{patient_id}.csv")
            patient_data.to_csv(path, sep=";", index=False)
            path_list.append(path)
        all_descriptions = pd.concat(descriptions)
        descriptions_file_path = os.path.join(self.results_path, "patients_descriptions.csv")
        all_descriptions.to_csv(descriptions_file_path, sep=";")
        # ChartGeneration("robust_scaling").plot_patient_data(scaled_patients)
        return path_list

    @staticmethod
    def polyfit_baseline(y, degree):
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        return baseline
