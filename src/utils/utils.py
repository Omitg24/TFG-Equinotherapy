import os
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# GENERAL FUNCTIONS
def str_to_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")


def timestamp_to_date(timestamp: int) -> str:
    return datetime.fromtimestamp((int(timestamp) / 1000)).strftime("%Y-%m-%d %H:%M:%S.%f")


def calculate_magnitude(accx_values, accy_values, accz_values):
    return np.sqrt(accx_values ** 2 + accy_values ** 2 + accz_values ** 2)


def unify_sleep_stages(data: pd.DataFrame, n_phases) -> pd.DataFrame:
    if n_phases != 5:
        if n_phases == 2:
            data.loc[data['stage'] != 'Wake', 'stage'] = 'Sleep'
        elif n_phases == 3:
            data.loc[data['stage'].isin(['N1', 'N2', 'N3']), 'stage'] = 'NREM'
        elif n_phases == 4:
            data.loc[data['stage'].isin(['N1', 'N2']), 'stage'] = 'LN'
        else:
            raise ValueError("Number of phases must be between 2 and 5")
    return data


# TIME WINDOWS FUNCTIONS
def create_time_windows(data: pd.DataFrame, w_size: float, w_overlapping: float) -> List[pd.DataFrame]:
    start_time = str_to_date(data.iloc[0]["date"])
    end_time = start_time + timedelta(seconds=w_size)
    next_window_start_time = end_time - timedelta(seconds=w_overlapping)
    data_windows = []

    index = next_window_index = 0
    current_window = []
    found_first_index = False

    while index < len(data):
        current_time = str_to_date(data.iloc[index]["date"])
        if current_time <= end_time:
            if next_window_start_time <= current_time and not found_first_index:
                next_window_index = index
                found_first_index = True
                start_time = current_time
            current_window.append(data.iloc[index].tolist())
            index += 1
        else:
            if next_window_start_time == end_time:
                next_window_index = index
                start_time = current_time
            data_windows.append(pd.DataFrame(current_window, columns=data.columns))
            current_window = []
            end_time = start_time + timedelta(seconds=w_size)
            next_window_start_time = end_time - timedelta(seconds=w_overlapping)
            index = next_window_index
            found_first_index = False

    return data_windows


# LUIGI LOCAL TARGET FUNCTIONS
def create_output_paths_file(output_path: str, path_list: List[str]):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "output_paths.txt"), "w") as file:
        for path in path_list:
            file.write(f"{path}\n")


def get_prev_files_path(path: str):
    paths = []
    with open(path, "r") as file:
        for line in file:
            paths.append(line.strip())
    return paths


def get_partitions(partitions_files: List[str]) -> Dict:
    partitions = {}

    for partition_file in partitions_files:
        partition_type = os.path.dirname(partition_file).split("/")[-1]
        if partition_type not in partitions.keys():
            partitions[partition_type] = {}
        filename = os.path.basename(partition_file).split(".")[0]
        partitions[partition_type][filename] = pd.read_csv(partition_file, sep=";")
    return partitions


# PLOTTING FUNCTIONS
def plot_model(data: pd.DataFrame, model: np.ndarray):
    plt.scatter(data['secs'], data['ppm'], color='black')  # Puntos negros
    plt.plot(data['secs'], np.polyval(model, np.log(data['secs'])), color='black',
             linestyle='--')  # Línea discontinua negra
    plt.xlabel('Seconds')
    plt.ylabel('Beats per minute')
    plt.legend(['Model', 'Data'], loc='upper left')  # Ajusta la leyenda
    plt.show()


def plot_hr_variation(hr_list: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(hr_list, color='tab:blue')
    plt.title('Heart Rate Plot')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate')
    plt.grid(True)
    plt.show()
    plt.close()


def plot_baseline(data: pd.DataFrame, columns: List[str], patient_id: str):
    path = f"files/img/charts/{patient_id}/"
    os.makedirs(path, exist_ok=True)
    for column in columns:
        baseline_column = f"{column}_baseline"
        plt.figure(figsize=(10, 6))
        plt.plot(data[column], label=column, color="blue", alpha=0.75)
        plt.plot(data[baseline_column], label=f'{column} Baseline', color="tab:orange", linestyle='solid', linewidth=2)
        plt.title(f'{column} and its Baseline')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        plt.savefig(f'{path}{column}_baseline.png')
        plt.close()


def plot_boxplot(data: List[pd.DataFrame], column: str):
    os.makedirs("files/img/charts/", exist_ok=True)
    plt.figure(figsize=(14, 8))
    column_data = [df[column] for df in data]
    plt.boxplot(column_data, widths=0.6, patch_artist=True)
    plt.title(f'Boxplot of {column.upper()}', fontsize=16)
    plt.ylabel('Value', fontsize=14)
    plt.xlabel('Patient', fontsize=14)
    plt.xticks(ticks=range(1, len(data) + 1), labels=[f'Patient S{i + 1}' for i in range(len(data))], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("files/img/charts/", f"boxplot_{column}.png"))
    plt.close()


def plot_confusion_matrix(confusion_matrix, model_name, classes, save_path):
    with np.errstate(all='ignore'):
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    fmt = '.2%'

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig(save_path)
    plt.close()


def generate_comparison_plots(adjusted: List[pd.DataFrame], min_max_scaled: List[pd.DataFrame],
                              robust_scaled: List[pd.DataFrame]):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Comparison of Original, Min-Max Scaled, and Robust Scaled Data')

    adjusted_concat = pd.concat(adjusted)
    min_max_scaled_concat = pd.concat(min_max_scaled)
    robust_scaled_concat = pd.concat(robust_scaled)

    adjusted_concat['magnitude'] = calculate_magnitude(adjusted_concat['accx'], adjusted_concat['accy'],
                                                       adjusted_concat['accz'])
    min_max_scaled_concat['magnitude'] = calculate_magnitude(min_max_scaled_concat['accx'],
                                                             min_max_scaled_concat['accy'],
                                                             min_max_scaled_concat['accz'])
    robust_scaled_concat['magnitude'] = calculate_magnitude(robust_scaled_concat['accx'], robust_scaled_concat['accy'],
                                                            robust_scaled_concat['accz'])

    sns.scatterplot(x='magnitude', y='hr', data=adjusted_concat, ax=axs[0, 0])
    sns.histplot(adjusted_concat['magnitude'], ax=axs[0, 1], kde=True)
    axs[0, 0].set_title('Original Data')
    axs[0, 1].set_title('Original Data Distribution')

    sns.scatterplot(x='magnitude', y='hr', data=min_max_scaled_concat, ax=axs[1, 0])
    sns.histplot(min_max_scaled_concat['magnitude'], ax=axs[1, 1], kde=True)
    axs[1, 0].set_title('Min-Max Scaled Data')
    axs[1, 1].set_title('Min-Max Scaled Data Distribution')

    sns.scatterplot(x='magnitude', y='hr', data=robust_scaled_concat, ax=axs[2, 0])
    sns.histplot(robust_scaled_concat['magnitude'], ax=axs[2, 1], kde=True)
    axs[2, 0].set_title('Robust Scaled Data')
    axs[2, 1].set_title('Robust Scaled Data Distribution')

    ax = axs[0, 0]
    for ax in axs[:, 0]:
        ax.set_xlabel('Magnitude of Acceleration')
    ax.set_ylabel('Heart Rate')
    for ax in axs[:, 1]:
        ax.set_xlabel('Magnitude of Acceleration')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    plt.savefig('files/img/charts/scaling_comparison.png')


def generate_combined_boxplot_comparison(fold_data, loo_data, path):
    results = []

    for key, value in fold_data.items():
        model_name = key[0]
        f1_score = value['f1_score']
        results.append([model_name, "stratified_cv", f1_score])

    for key, value in loo_data.items():
        model_name = key[0]
        f1_score = value['f1_score']
        results.append([model_name, "leave_one_participant", f1_score])

    df = pd.DataFrame(results, columns=['Model', 'ValidationMethod', 'F1Score'])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='F1Score', hue='ValidationMethod', data=df)
    plt.title('Comparative Boxplot for Different Validation Methods')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'boxplot_comparison_combined.png'))
    plt.show()
