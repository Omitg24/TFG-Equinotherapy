import luigi


class ProjectConfig(luigi.Config):
    # Patient stages and watches data path
    patient_data_path = luigi.Parameter()
    # Preprocessing pipeline
    consolidation_path = luigi.Parameter()
    cleaning_path = luigi.Parameter()
    scaling_path = luigi.Parameter()
    features_path = luigi.Parameter()
    # Shallow training pipeline
    shallow_partitions_path = luigi.Parameter()
    shallow_oversampling_path = luigi.Parameter()
    shallow_training_path = luigi.Parameter()
    shallow_analysis_path = luigi.Parameter()
    # Deep training pipeline
    deep_partitions_path = luigi.Parameter()
    deep_training_path = luigi.Parameter()
    deep_analysis_path = luigi.Parameter()
    # Window creation
    w_size = luigi.IntParameter()
    w_overlapping = luigi.IntParameter()
    w_min_duration = luigi.IntParameter()
    sample_frequency = luigi.IntParameter()
    # Partitioning configuration
    n_phases = luigi.IntParameter()
    n_splits = luigi.IntParameter()
    test_percentage = luigi.FloatParameter()
    # Deep training configuration
    lstm_mode = luigi.IntParameter()
    neural_network = luigi.IntParameter()
    n_epochs = luigi.IntParameter()
