import os

import luigi
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, \
    precision_score
from tensorflow.python.keras.models import load_model

from . import DeepTraining
from .data_generator import DataGenerator
from ..utils import ProjectConfig, utils


class DeepAnalysis(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_path = ProjectConfig().deep_analysis_path
        self.last_epoch = (ProjectConfig().n_epochs - 1)

    def requires(self):
        return DeepTraining()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.results_path, "results.csv"))

    def run(self):
        prev_files = utils.get_prev_files_path(self.input().path)
        training_results = self._get_training_results(prev_files)
        self._test_experiments(training_results)

    def _test_experiments(self, training_results):
        results = {}
        for participant, participant_results in training_results.items():
            patient_idx = participant.split("_")[-1]
            model_path = participant_results[f"model_{patient_idx}"]
            test_set = participant_results[f"test_{patient_idx}"]
            weights_path = participant_results[f"epoch_{self.last_epoch}"][
                f"weights_{patient_idx}_{self.last_epoch}.weights"]
            model = load_model(model_path)
            model.load_weights(weights_path)
            results[participant] = self._test_model(model, test_set, participant)
        self._save_results(results)

    def _test_model(self, model, test_set, participant):
        X_test, y_test = (test_set[['accx', 'accy', 'accz', 'hr']], test_set['stage'])
        eval_generator = DataGenerator(x_data=X_test,
                                       y_data=y_test,
                                       name='test',
                                       window_size=ProjectConfig().w_size,
                                       window_overlap=ProjectConfig().w_overlapping,
                                       lstm_mode=ProjectConfig().lstm_mode,
                                       n_clases=ProjectConfig().n_phases,
                                       sample_frequency=ProjectConfig().sample_frequency,
                                       is_training=False)
        class_labels = y_test.unique()
        return self._evaluate(model, eval_generator, class_labels, participant)

    def _evaluate(self, model, eval_generator, class_labels, participant):
        y_pred = []
        y_true = []
        for i in range(eval_generator.__len__()):
            batch_x, batch_y = eval_generator.__getitem__(i)
            batch_pred = model.predict(batch_x)
            y_pred.extend(np.argmax(batch_pred, axis=1))
            y_true.extend(np.argmax(batch_y, axis=1))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn = conf_matrix.sum(axis=1) - conf_matrix.sum(axis=0) + conf_matrix.diagonal()
        fp = conf_matrix.sum(axis=0) - conf_matrix.diagonal()
        specificity = tn / (tn + fp)
        specificity_weighted = np.average(specificity, weights=np.bincount(y_true))
        class_report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0)

        print(f"Participant: {participant}")
        print(class_report)

        save_dir = os.path.join(self.results_path, participant)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "confusion_matrix.png")

        utils.plot_confusion_matrix(conf_matrix, "LSTM", class_labels, save_path)
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity_weighted
        }
        return results

    def _get_training_results(self, prev_files):
        training_results = {}
        last_epoch_str = f"epoch_{self.last_epoch}"

        for file in prev_files:
            parts = os.path.normpath(file).split(os.sep)
            patient_idx = parts[4]

            if "epoch" in file:
                epoch_part = parts[5] if last_epoch_str in file else None
                if epoch_part:
                    file_part = parts[6]
                    file_name, ext = os.path.splitext(file_part)
                    content = pd.read_csv(file, sep=";") if ext == ".csv" else file
                    training_results.setdefault(patient_idx, {}).setdefault(epoch_part, {})[file_name] = content
            else:
                file_part = parts[5]
                file_name, ext = os.path.splitext(file_part)
                content = pd.read_csv(file, sep=";") if ext == ".csv" else file
                training_results.setdefault(patient_idx, {})[file_name] = content
        return training_results

    def _save_results(self, results):
        with open(os.path.join(self.results_path, "results.csv"), "w") as f:
            f.write("participant;accuracy;precision;recall;f1;specificity\n")
            for participant, participant_results in results.items():
                f.write(
                    f"{participant};{participant_results['accuracy']};{participant_results['precision']};"
                    f"{participant_results['recall']};{participant_results['f1']};{participant_results['specificity']}"
                    f"\n")
        print("Results saved successfully")
