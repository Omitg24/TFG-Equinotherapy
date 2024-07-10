import os.path
from statistics import mode

import luigi
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..utils import ProjectConfig, utils
from . import Scaling
from .features import *


class Preprocessing(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().features_path
        self.current_patient = None
        self.features_labels = ["HR_MEAN", "HR_SD", "AOM_MEAN", "AOM_SD", "ANGLE_MEAN", "ANGLE_SD",
                                "X_MEAN", "X_SD", "X_CORR", "X_KURT", "X_CREST", "X_SKEW", "X_ZERO", "X_ENTR", "X_ENER",
                                "X_FLUX",
                                "Y_MEAN", "Y_SD", "Y_CORR", "Y_KURT", "Y_CREST", "Y_SKEW", "Y_ZERO", "Y_ENTR", "Y_ENER",
                                "Y_FLUX",
                                "Z_MEAN", "Z_SD", "Z_CORR", "Z_KURT", "Z_CREST", "Z_SKEW", "Z_ZERO", "Z_ENTR", "Z_ENER",
                                "Z_FLUX",
                                "MAGACC_MEAN", "MAGACC_SD", "MAGACC_KURT", "MAGACC_CREST", "MAGACC_SKEW", "MAGACC_ZERO",
                                "MAGACC_ENTR", "MAGACC_ENER", "MAGACC_FLUX"]

    def requires(self):
        return Scaling()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        prev_files = utils.get_prev_files_path(self.input().path)

        features_matrix_list = []
        for prev_file in prev_files:
            patient_data = pd.read_csv(prev_file, sep=";")
            self.current_patient = patient_data["patient_id"].iloc[0]
            print(f"Patient: {self.current_patient}")
            features_matrix_list.append(self._process_file(patient_data))
        path_list = self._select_most_important_features(features_matrix_list)
        utils.create_output_paths_file(self.results_path, path_list)

    def _process_file(self, data):
        consecutive_windows = self._create_consecutive_windows(data,
                                                               ProjectConfig().w_min_duration)
        time_windows = self._create_time_windows(consecutive_windows)
        data_dict = self._create_dict(time_windows)
        features_matrix = self._extract_features(data_dict)
        scaler = StandardScaler()
        features_matrix_scaled = scaler.fit_transform(features_matrix.iloc[:, :-1])
        features_matrix_scaled_df = pd.DataFrame(features_matrix_scaled, columns=features_matrix.columns[:-1])
        features_matrix_scaled_df['stage'] = features_matrix['stage'].values
        features_matrix_scaled_df['patient_id'] = self.current_patient
        self._calculate_stages_ratio(features_matrix_scaled_df['stage'].astype(str).tolist())
        return features_matrix_scaled_df

    def _select_most_important_features(self, features_matrix_list):
        all_features_df = pd.concat(features_matrix_list, axis=0)

        X = all_features_df.drop(columns=['stage', 'patient_id'])
        y = all_features_df['stage']

        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)

        p_values = selector.pvalues_
        selected_features = X.columns[p_values < 0.005]

        final_features_df = all_features_df[selected_features].copy()
        final_features_df['stage'] = all_features_df['stage']
        final_features_df['patient_id'] = all_features_df['patient_id']

        patients = final_features_df['patient_id'].unique()
        path_list = []
        for patient in patients:
            patient_data = final_features_df[final_features_df['patient_id'] == patient]
            path = os.path.join(self.results_path, f"features_{patient}.csv")
            patient_data.to_csv(path, sep=";", index=False)
            path_list.append(path)
        return path_list

    def _extract_features(self, data_windows):
        print("\tExtracting features...")
        feature_list = self.features_labels + ["stage"]

        # Crear un DataFrame vacío con las columnas de feature_list
        features_data = pd.DataFrame(columns=feature_list)

        for data_idx in tqdm(range(len(data_windows))):
            data_row_hr = np.array(data_windows[data_idx]["hr"])
            data_row_accx = np.array(data_windows[data_idx]["accx"])
            data_row_accy = np.array(data_windows[data_idx]["accy"])
            data_row_accz = np.array(data_windows[data_idx]["accz"])
            data_row_magacc = np.array(data_windows[data_idx]["magacc"])
            data_row_stage = np.array(data_windows[data_idx]["stage"])

            aom_vals = aom3_seconds(data_row_magacc)
            angle_vals = angle3_seconds(data_row_accx, data_row_accy, data_row_accz)

            feature_dict = {"stage": mode(data_row_stage),
                            "HR_MEAN": np.mean(data_row_hr), "HR_SD": np.std(data_row_hr),
                            "AOM_MEAN": np.mean(aom_vals), "AOM_SD": np.std(aom_vals),
                            "ANGLE_MEAN": np.mean(angle_vals), "ANGLE_SD": np.std(angle_vals),
                            # X features
                            "X_MEAN": np.mean(data_row_accx),
                            "X_SD": np.std(data_row_accx),
                            "X_CORR": feature_correlation(data_row_accx, data_row_accy),
                            "X_KURT": feature_kurtosis(data_row_accx),
                            "X_CREST": feature_crest_factor(data_row_accx),
                            "X_SKEW": feature_skewness(data_row_accx),
                            "X_ZERO": feature_zero_crossing(data_row_accx),
                            "X_ENTR": feature_entropy(data_row_accx),
                            "X_ENER": feature_band_energy(data_row_accx, 10),
                            "X_FLUX": feature_spectral_flux(data_row_accx),
                            # Y features
                            "Y_MEAN": np.mean(data_row_accy),
                            "Y_SD": np.std(data_row_accy),
                            "Y_CORR": feature_correlation(data_row_accy, data_row_accz),
                            "Y_KURT": feature_kurtosis(data_row_accy),
                            "Y_CREST": feature_crest_factor(data_row_accy),
                            "Y_SKEW": feature_skewness(data_row_accy),
                            "Y_ZERO": feature_zero_crossing(data_row_accy),
                            "Y_ENTR": feature_entropy(data_row_accy),
                            "Y_ENER": feature_band_energy(data_row_accy, 10),
                            "Y_FLUX": feature_spectral_flux(data_row_accy),
                            # Z features
                            "Z_MEAN": np.mean(data_row_accz),
                            "Z_SD": np.std(data_row_accz),
                            "Z_CORR": feature_correlation(data_row_accz, data_row_accx),
                            "Z_KURT": feature_kurtosis(data_row_accz),
                            "Z_CREST": feature_crest_factor(data_row_accz),
                            "Z_SKEW": feature_skewness(data_row_accz),
                            "Z_ZERO": feature_zero_crossing(data_row_accz),
                            "Z_ENTR": feature_entropy(data_row_accz),
                            "Z_ENER": feature_band_energy(data_row_accz, 10),
                            "Z_FLUX": feature_spectral_flux(data_row_accz),
                            # MAGACC features
                            "MAGACC_MEAN": np.mean(data_row_magacc),
                            "MAGACC_SD": np.std(data_row_magacc),
                            "MAGACC_KURT": feature_kurtosis(data_row_magacc),
                            "MAGACC_CREST": feature_crest_factor(data_row_magacc),
                            "MAGACC_SKEW": feature_skewness(data_row_magacc),
                            "MAGACC_ZERO": feature_zero_crossing(data_row_magacc),
                            "MAGACC_ENTR": feature_entropy(data_row_magacc),
                            "MAGACC_ENER": feature_band_energy(data_row_magacc, 10),
                            "MAGACC_FLUX": feature_spectral_flux(data_row_magacc)}

            features_data.loc[data_idx] = feature_dict

        return features_data

    @staticmethod
    def _calculate_stages_ratio(stages):
        stage_counts = {}
        total = len(stages)
        for stage in stages:
            if stage in stage_counts:
                stage_counts[stage] += 1
            else:
                stage_counts[stage] = 1
        ratio_str = ", ".join(f"{stage}: {count / total * 100:.2f}%" for stage, count in stage_counts.items())
        print(f"\t{ratio_str}")

    @staticmethod
    def _create_dict(time_windows):
        data_dict = []
        for time_window in time_windows:
            window_dict = {
                "accx": time_window["accx"].astype(float).tolist(),
                "accy": time_window["accy"].astype(float).tolist(),
                "accz": time_window["accz"].astype(float).tolist(),
                "magacc": time_window["magacc"].astype(float).tolist(),
                "hr": time_window["hr"].astype(float).tolist(),
                "stage": time_window["stage"].astype(str).tolist()
            }
            data_dict.append(window_dict)
        return data_dict

    @staticmethod
    def _create_time_windows(consecutive_windows):
        time_windows = []
        for c_window in consecutive_windows:
            time_windows.extend(
                utils.create_time_windows(data=c_window, w_size=ProjectConfig().w_size,
                                          w_overlapping=ProjectConfig().w_overlapping))
        return time_windows

    @staticmethod
    def _create_consecutive_windows(data, window_duration):
        data_windows, current_window = [], []
        prev_time = None

        for index in range(len(data)):
            current_time = utils.str_to_date(data.iloc[index]["date"])
            if prev_time is not None and (current_time - prev_time).seconds > 1:
                start_time = utils.str_to_date(current_window[0]["date"])
                if (prev_time - start_time).seconds >= window_duration:
                    data_windows.append(pd.DataFrame(current_window, columns=data.columns))
                current_window = []
            else:
                current_window.append(data.iloc[index].to_dict())
            prev_time = utils.str_to_date(data.iloc[index]["date"])
        print(f"\tFound a total of {len(data_windows)} consecutive windows")
        return data_windows
