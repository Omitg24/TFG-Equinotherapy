import os

import joblib
import luigi
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..utils import ProjectConfig, utils
from .training import ShallowTraining


class ShallowAnalysis(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().shallow_analysis_path
        self.partitions_path = ProjectConfig().shallow_partitions_path

    def requires(self):
        return ShallowTraining()

    def output(self):
        output = [luigi.LocalTarget(os.path.join(self.results_path, "stratified_cv_best_models.csv")),
                  luigi.LocalTarget(os.path.join(self.results_path, "leave_one_participant_best_models.csv"))]
        return output

    def run(self):
        # model;validation_method;partition;param_idx;param_combinations;accuracy;f1_score;model_path
        model_info_path = self.input().path
        models_info = pd.read_csv(model_info_path, sep=";")
        fold_best_models = self._get_best_models(models_info, "fold")
        loo_best_models = self._get_best_models(models_info, "leave_one_participant")
        print("Stratified KFold")
        self._show_best_models(fold_best_models)
        self._save_best_models(fold_best_models, "stratified_cv")
        self._calculate_confusion_matrix(fold_best_models)
        print("Leave One Participant Out")
        self._show_best_models(loo_best_models)
        self._save_best_models(loo_best_models, "leave_one_participant")
        self._calculate_confusion_matrix(loo_best_models)

        utils.generate_combined_boxplot_comparison(fold_best_models, loo_best_models, self.results_path)

    def _calculate_confusion_matrix(self, data):
        for key, value in tqdm(data.items()):
            partition_type = value["validation_method"]
            model_path = value["model_path"]
            model = joblib.load(model_path)
            partition_index = int(key[1].split("_")[-1])

            X_val, y_val, label_encoder = self._load_partitions(partition_type, partition_index)

            predictions = model.predict(X_val)
            cm = confusion_matrix(y_val, predictions)

            save_dir = os.path.join(self.results_path, "confusion_matrix", partition_type, key[0])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"confusion_matrix_{key[0]}_{key[1]}.png")

            utils.plot_confusion_matrix(cm, key, label_encoder.classes_, save_path)

    def _load_partitions(self, partition_type, partition_index):
        partition_file_path = os.path.join(self.partitions_path, partition_type,
                                           f"validation_{partition_type}_{partition_index}.csv")
        validation_data = pd.read_csv(partition_file_path, sep=";")

        label_encoder = LabelEncoder()
        label_encoder.fit(validation_data['stage'])

        X_validation = validation_data.drop(columns=['stage'])
        y_validation = label_encoder.transform(validation_data['stage'])

        return X_validation, y_validation, label_encoder

    def _save_best_models(self, data, file_name):
        os.makedirs(self.results_path, exist_ok=True)
        partitions = sorted(set(key[1] for key in data.keys()))
        label_models = set(key[0] for key in data.keys())

        with open(f"{self.results_path}{file_name}_best_models.csv", 'w') as file:
            file.write("partition;" + ";".join(label_models) + "\n")
            for partition in partitions:
                row_data = [partition] + [str(data.get((model, partition))["f1_score"]) for model in label_models]
                file.write(";".join(row_data) + "\n")
        print(f"\tFile saved in '{self.results_path}{file_name}_best_models.csv'")

    @staticmethod
    def _show_best_models(data):
        for key, value in data.items():
            print(f"\t * {key} - ACC: {value['accuracy']:.2f} F1:{value['f1_score']:.2f}")
            print(f"\t\t{value['model_path']}")

    @staticmethod
    def _get_best_models(models_info, validation_method):
        models_filtered = models_info[models_info["validation_method"] == validation_method]
        data = {}

        for _, row in models_filtered.iterrows():
            key = (row["model"], f"{validation_method}_{row['partition']}")

            current_entry = {
                "validation_method": row["validation_method"],
                "param_idx": row["param_idx"],
                "param_combinations": row["param_combinations"],
                "accuracy": row["accuracy"],
                "f1_score": row["f1_score"],
                "model_path": row["model_path"]
            }

            data[key] = current_entry

        return data
