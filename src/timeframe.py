import argparse
import os
from typing import List

import pandas as pd

from chart_generation import ChartGeneration
from utils import utils


class TimeFrame:
    """
    Extracts the start and end timestamps from the watches data and saves it in a csv file.
    It also generates a chart for each patient with its ACC Magnitude and Heart Rate.
    """

    def __init__(self):
        self.results_path = "files/img/charts/"
        self.final_file = ["patient;start_timestamp;start_time;end_timestamp;end_time"]

    def run(self, watches_directory: str):
        watches_folders = sorted(os.listdir(watches_directory))
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        for patient_id in watches_folders:
            patient_watches_folder = os.path.join(watches_directory, patient_id)

            patient_watches_directory = os.listdir(patient_watches_folder)

            print(f"Patient: {patient_id}")
            if len(patient_watches_directory) == 2:
                first_half = pd.read_csv(os.path.join(watches_directory, patient_id, patient_watches_directory[0]))
                second_half = pd.read_csv(os.path.join(watches_directory, patient_id, patient_watches_directory[1]))
                watch_data = pd.concat([first_half, second_half.iloc[1:]]).reset_index(drop=True)

                ChartGeneration("original").generate_plot(watch_data, patient_id)
                start_timestamp, end_timestamp = self._get_timeframe(watch_data)
                start_time = utils.timestamp_to_date(start_timestamp)
                end_time = utils.timestamp_to_date(end_timestamp)
                print(f"\tStart: {start_time} - End: {end_time}")
                self.final_file.append(f"{patient_id};{start_timestamp};{start_time};{end_timestamp};{end_time}")
            else:
                print("\tNot enough files in watches folder")
        self._write_to_file()

    def _write_to_file(self):
        with open("files/timeframe.csv", "w") as file:
            for line in self.final_file:
                file.write(line + "\n")
        print(f"File saved in 'files/timeframe.csv'")

    @staticmethod
    def convert_to_dataframe(watch_data: List[str]) -> pd.DataFrame:
        headers = watch_data[0].split(";")
        data = [row.split(";") for row in watch_data[1:] if len(row) > 1]
        df = pd.DataFrame(data, columns=headers)
        return df

    @staticmethod
    def _get_timeframe(watch_data: pd.DataFrame) -> (int, int):
        timestamps = watch_data["time"]
        return min(timestamps), max(timestamps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Preprocessing',
                                     description='Data framing getting the start and end timestamps and generating '
                                                 'acceleration magnitude and heart rate charts.')
    parser.add_argument('watches_directory', type=str, help='Directory containing watches data')
    args = parser.parse_args()

    print("Processing all patient data...")
    timeframe = TimeFrame()
    timeframe.run(args.watches_directory)
