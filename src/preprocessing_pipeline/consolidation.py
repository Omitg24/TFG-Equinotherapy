import os

import luigi
import pandas as pd
from tqdm import tqdm

from src.utils import ProjectConfig, utils


class Consolidation(luigi.Task):
    """
    Consolidation of the data gathered from the wearable devices and the stages to create a final file with the
    synchronized information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = ProjectConfig().patient_data_path
        self.results_path = ProjectConfig().consolidation_path
        self.current_patient = None

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "output_paths.txt"))

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        stages_path = os.path.join(str(self.data_path), "stages")
        watches_path = os.path.join(str(self.data_path), "watches")

        stages_files = sorted([".".join(f.split(".")[:-1]) for f in os.listdir(stages_path) if f.endswith(".csv")])
        watches_folders = sorted(os.listdir(watches_path))

        common_patients = [patient for patient in stages_files if patient in watches_folders]
        paths = []
        for patient_id in common_patients:
            self.current_patient = patient_id
            patient_path = self._process_patient_data(stages_path, watches_path)
            paths.append(patient_path)
        utils.create_output_paths_file(self.results_path, paths)

    def _process_patient_data(self, stages_path, watches_path):
        patient_stages_file = f"{self.current_patient}.csv"
        patient_watches_folder = os.path.join(watches_path, self.current_patient)

        patient_watches_directory = sorted(os.listdir(patient_watches_folder))

        print(f"Patient: {self.current_patient}")
        if len(patient_watches_directory) == 2:
            first_half = pd.read_csv(os.path.join(watches_path, self.current_patient, patient_watches_directory[0]),
                                     sep=";")
            second_half = pd.read_csv(os.path.join(watches_path, self.current_patient, patient_watches_directory[1]),
                                      sep=";")
            watch_data = pd.concat([first_half, second_half.iloc[1:]]).reset_index(drop=True)
            stage_data = pd.read_csv(os.path.join(stages_path, patient_stages_file), sep=";")
            return self._process_files(watch_data, stage_data)
        else:
            print("\tNot enough files in watches folder")

    def _process_files(self, watch_data, stage_data):
        print("\tProcessing files...")
        start_time, end_time, stages = self._process_stages(stage_data)
        start_index, end_index = self._process_watch(watch_data, start_time, end_time)
        processed_file = self._create_final_file(watch_data, stages, start_index, end_index)
        path = os.path.join(self.results_path, f"consolidated_{self.current_patient}.csv")
        processed_file.to_csv(path, sep=";", index=False)
        print(f"\tFile saved in {path}")
        return path

    @staticmethod
    def _create_final_file(watch_data, stages, start_index, end_index) -> pd.DataFrame:
        # Create a DataFrame with the final data
        final_data = watch_data.iloc[start_index:end_index + 1].copy()
        final_data['stage'] = 'ND'  # Initialize with 'ND'
        frequency = (end_index - start_index) / len(stages)
        stage_index, frequency_counter = 0, 1

        for line_idx in tqdm(range(start_index, end_index + 1)):
            if stage_index < len(stages):
                final_data.at[line_idx, 'stage'] = stages[stage_index]
                if frequency <= frequency_counter:
                    stage_index += 1
                    frequency_counter = 0
                frequency_counter += 1

        date_col = final_data.iloc[:, 0].apply(lambda x: utils.timestamp_to_date(x))
        final_data.insert(0, 'date', date_col)

        return final_data

    @staticmethod
    def _process_watch(watch_data, start_time, end_time):
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)

        timestamps = watch_data["time"]

        start_index, end_index, counter = None, None, 1

        for timestamp in timestamps:
            if timestamp >= start_timestamp and start_index is None:
                start_index = counter
            if end_timestamp <= timestamp and end_index is None:
                end_index = counter
            counter += 1

        if end_index is None:
            end_index = len(timestamps) - 1
        return start_index, end_index

    @staticmethod
    def _process_stages(stage_data):
        start_time = utils.str_to_date(stage_data.iloc[0, 2])
        end_time = utils.str_to_date(stage_data.iloc[0, 3])
        stages = stage_data["stage"].values.tolist()
        return start_time, end_time, stages
