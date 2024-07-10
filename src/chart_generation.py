import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ChartGeneration:
    def __init__(self, data_type: str):
        self.data_type = data_type
        self.results_path = os.path.join("files", "img", "charts")

    def run(self, directory: str):
        files = [f for f in os.listdir(directory) if f.endswith(".csv")]
        for file in files:
            patient_id = file.split(".")[0].split("_")[1]
            data = pd.read_csv(os.path.join(directory, file), sep=";")
            self.generate_plot(data, patient_id)

    def generate_plot(self, data: pd.DataFrame, patient_id: str, mag_max_limit=60, hr_max_limit=140, mag_min_limit=0,
                      hr_min_limit=0):
        path = os.path.join(self.results_path, patient_id)
        os.makedirs(path, exist_ok=True)
        print("\tGenerating acceleration magnitude and heart rate chart...")
        datetime_timestamps = data["time"].astype(np.int64).tolist()
        accx = data["accx"].astype(float).tolist()
        accy = data["accy"].astype(float).tolist()
        accz = data["accz"].astype(float).tolist()
        hr = data["hr"].astype(float).tolist()

        magnitude = self._calculate_magnitude(accx, accy, accz) if "magacc" not in data.columns else data[
            "magacc"].astype(float).tolist()

        mean_hr = np.mean(hr)
        mean_magnitude = np.mean(magnitude)
        print(f"\t - Mean Heart Rate: {mean_hr}")
        print(f"\t - Mean Magnitude: {mean_magnitude}")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Time')
        ax1.set_ylabel('ACC Magnitude')
        ax1.plot(datetime_timestamps, magnitude, color='tab:green')
        ax1.tick_params(axis='y')
        ax1.set_ylim(bottom=mag_min_limit, top=mag_max_limit)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Heart Rate')
        ax2.plot(datetime_timestamps, hr, color='tab:red')
        ax2.tick_params(axis='y')
        ax2.set_ylim(bottom=hr_min_limit, top=hr_max_limit)

        fig.tight_layout()

        plt.title(f'{self.data_type}_{patient_id} - ACC Magnitude and Heart Rate')
        plt.grid(True)

        ax1.legend(['ACC Magnitude'], loc='upper left')
        ax2.legend(['Heart Rate'], loc='upper right')
        info_text = f"Mean Heart Rate: {mean_hr:.2f}\nMean Magnitude: {mean_magnitude:.2f}"
        ax1.text(0.5, 0.95, info_text, transform=ax1.transAxes, fontsize=10, verticalalignment='center',
                 horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

        print(f"\tChart saved in '{path}/{self.data_type}_chart.png'")
        plt.savefig(f'{path}/{self.data_type}_chart.png')
        plt.close()

    def plot_patient_data(self, dataframes: List[pd.DataFrame]):
        fig_acc, ax_acc = plt.subplots(figsize=(14, 7))
        fig_hr, ax_hr = plt.subplots(figsize=(14, 7))
        for df in dataframes:
            patient_id = df["patient_id"].iloc[0]
            df['mag_acc'] = self._calculate_magnitude(df['accx'], df['accy'], df['accz'])
            ax_acc.plot(df.index, df['mag_acc'], label=f'Patient {patient_id}')
            ax_hr.plot(df.index, df['hr'], label=f'Patient {patient_id}')
        ax_acc.set_title('Magnitude of Acceleration for All Patients')
        ax_acc.set_ylabel('Magnitude of Acceleration')
        ax_acc.set_xlabel('Time')
        ax_acc.legend()

        acc_plot_path = os.path.join(self.results_path, f"{self.data_type}_global_magacc_plot.png")
        fig_acc.savefig(acc_plot_path)
        print(f"Saved acceleration magnitude plot to {acc_plot_path}")

        ax_hr.set_title('Pulse (Heart Rate) for All Patients')
        ax_hr.set_ylabel('Pulse (HR)')
        ax_hr.set_xlabel('Time')
        ax_hr.legend()

        hr_plot_path = os.path.join(self.results_path, f"{self.data_type}_global_hr_plot.png")
        fig_hr.savefig(hr_plot_path)
        print(f"Saved pulse plot to {hr_plot_path}")

        plt.close(fig_acc)
        plt.close(fig_hr)

    @staticmethod
    def _calculate_magnitude(accx, accy, accz):
        magnitudes = []
        for x, y, z in zip(accx, accy, accz):
            magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            magnitudes.append(magnitude)
        return magnitudes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Preprocessing',
                                     description='Acceleration magnitude and heart rate chart generation')
    parser.add_argument('data_type', type=str, help='Type of data to be processed based on its phase')
    parser.add_argument('directory', type=str, help='Directory containing time series data')
    args = parser.parse_args()

    chart_generation = ChartGeneration(args.data_type)
    chart_generation.run(args.directory)
