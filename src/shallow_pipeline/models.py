import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

random_state = 0

# Modelos Shallow (Supervised)
shallow_models = {
    # STOCHASTIC
    "NB": BernoulliNB(),
    "SVC": SVC(random_state=random_state, probability=False),
    "RGC": RidgeClassifier(),
    # TREE-BASED
    "DT": DecisionTreeClassifier(random_state=random_state),
    # EMBED-BASED
    "RF": RandomForestClassifier(random_state=random_state),
    "ABC": AdaBoostClassifier(random_state=random_state),
    "CB": CatBoostClassifier(random_state=random_state, verbose=False, logging_level=None),
    "XGB": XGBClassifier(random_state=random_state, metric='multiclass', eval_metric='mlogloss', verbosity=0),
    "XT": ExtraTreesClassifier(random_state=random_state),
    # NEURALNETWORK BASED
    "MLP": MLPClassifier(random_state=random_state),
    # DISTANCE BASED
    "KNN": KNeighborsClassifier()
}

# IN general in sklearn Parallel support, n_jobs = -1 n_jobsint, default=None The number of jobs to run in parallel.
# fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a
# joblib.parallel_backend context. -1 means using all processors.
models_gpu = {
    # STOCHASTIC
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
    # No GPU support in sklearn
    # No Parallel support
    "NB": BernoulliNB(),
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # No GPU support in sklearn
    # No Parallel support
    "SVC": SVC(random_state=random_state),
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV
    # No GPU support in sklearn
    # No Parallel support
    "RGC": RidgeClassifier(),

    # TREE-BASED
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn-tree-decisiontreeclassifier
    # No GPU support in sklearn
    # No Parallel support
    "DT": DecisionTreeClassifier(random_state=random_state),

    # EMBED-BASED
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier
    # No GPU support in sklearn
    # Parallel support, n_jobs = -1
    "RF": RandomForestClassifier(random_state=random_state, n_jobs=-1),
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn-ensemble-adaboostclassifier
    # No GPU support in sklearn
    # No Parallel support
    # "ABC": AdaBoostClassifier(random_state=random_state),
    # https://catboost.ai/en/docs/features/training-on-gpu
    # task_type="GPU", devices=0:get_gpu_device_count()
    "CB": CatBoostClassifier(random_state=random_state, verbose=False, logging_level=None, task_type="GPU"),
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html
    # tree_method='gpu_hist',
    "XGB": XGBClassifier(random_state=random_state, metric='multiclass', eval_metric='mlogloss', verbosity=0,
                         tree_method='gpu_hist'),
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn-ensemble-extratreesclassifier
    # No GPU support in sklearn
    # Parallel support, n_jobs = -1
    "XT": ExtraTreesClassifier(random_state=random_state, n_jobs=-1),

    # NEURALNETWORK BASED
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    # No GPU support in sklearn
    # No Parallel support
    "MLP": MLPClassifier(random_state=random_state),

    # DISTANCE BASED
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn-neighbors-kneighborsclassifier
    # No GPU support in sklearn
    # Parallel support, n_jobs = -1
    "KNN": KNeighborsClassifier(n_jobs=-1)
}
label_models = list(shallow_models.keys())

param_grid = {
    "NB": {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'fit_prior': [True, False]
    },
    "SVC": {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    },
    "RGC": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    "DT": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8, 16]
    },
    "RF": {
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=1000, num=3)],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [int(x) for x in np.linspace(10, 30, num=2)] + [None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    "ABC": {
        'n_estimators': [10, 50, 100, 500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
    },
    "CB": {
        'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    },
    "XGB": {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    },
    "XT": {
        "random_state": [0, 1, 2, 3, 4],
        "n_estimators": [320, 340, 360, 380],
        "max_depth": [30, 32, 34, 38]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': 10.0 ** -np.arange(1, 5),
        'max_iter': [1000]
    },
    "KNN": {
        'n_neighbors': [5, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
        'leaf_size': [5, 10],
    }
}
