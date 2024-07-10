import luigi

from src.preprocessing_pipeline import *
from src.shallow_pipeline import *
from src.deep_pipeline import *

if __name__ == '__main__':
    # Data preparation
    preprocessing_pipeline = [Consolidation(), Cleaning(), Scaling(), Preprocessing()]
    luigi.build(preprocessing_pipeline, local_scheduler=True)
    # Run shallow pipeline launching models such as Random Forest, SVM, KNN and XGBoost (see models.py)
    shallow_pipeline = [ShallowPartitioning(), ShallowOversampling(), ShallowTraining(), ShallowAnalysis()]
    luigi.build(shallow_pipeline, local_scheduler=True)
    # Run deep pipeline launching LSTM neural networks
    # deep_pipeline = [DeepPartitioning(), DeepTraining(), DeepAnalysis()]
    # luigi.build(deep_pipeline, local_scheduler=True)
