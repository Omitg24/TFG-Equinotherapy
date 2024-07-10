import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA


class FeaturePlotter:
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.output_dir = "files/img/features_plots/"

    def plot_features(self):
        for filename in os.listdir(self.features_dir):
            if filename.endswith(".csv"):
                current_patient = filename.split("/")[-1].split(".")[0].split("_")[
                    1]  # {path}/{prev_file}_SX.csv -> SX
                print(f"Patient: {current_patient}")
                output_patient_dir = os.path.join(self.output_dir, current_patient)

                if not os.path.exists(output_patient_dir):
                    os.makedirs(output_patient_dir)

                filepath = os.path.join(self.features_dir, filename)
                data = pd.read_csv(filepath, sep=';')
                self._generate_plots(data, current_patient, output_patient_dir)
                self._generate_pca_plot(data, current_patient)

    def _generate_pca_plot(self, data, current_patient):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data.drop(columns=['stage']))

        plt.figure(figsize=(8, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1])
        plt.title(f'PCA Plot - Patient ID: {current_patient}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.tight_layout()

        plt.xlim(-1.5, 2.5)
        plt.ylim(-1.5, 2.5)

        output_filepath = os.path.join(self.output_dir, f'{current_patient}_PCA_PLOT.png')
        plt.savefig(output_filepath)
        plt.close()

    @staticmethod
    def _generate_plots(data, current_patient, output_patient_dir):
        for column in tqdm(data.columns):
            plt.figure(figsize=(8, 6))
            plt.hist(data[column], bins=20, color='skyblue', edgecolor='black')
            plt.title(f'{column} - Patient ID: {current_patient}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()

            output_filepath = os.path.join(output_patient_dir, f'{column}_histogram.png')
            plt.savefig(output_filepath)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='FeaturePlotter', description='Generate plots for features data.')
    parser.add_argument('features_directory', type=str, help='Directory containing features data.')

    args = parser.parse_args()

    feature_plotter = FeaturePlotter(args.features_directory)
    feature_plotter.plot_features()
