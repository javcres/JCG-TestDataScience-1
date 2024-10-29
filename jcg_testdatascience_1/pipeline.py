from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import (AddMissingIndicator, CategoricalImputer,
                                       MeanMedianImputer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jcg_testdatascience_1.config.core import config
from jcg_testdatascience_1.processing.features import ReplaceZeroWithNone

# Pipeline Setup
heart_disease_pipe = Pipeline(
    [
        # Categorical vars
        (
            "missing_indicator_categorical",
            AddMissingIndicator(
                variables=list(config.pipeline_config.categorical_vars_with_missing)
            ),
        ),
        (
            "missing_imputation_new_label",
            CategoricalImputer(
                imputation_method="missing",
                variables=list(
                    config.pipeline_config.categorical_vars_to_inpute_with_new_label
                ),
            ),
        ),
        (
            "missing_imputation_most_freq",
            CategoricalImputer(
                imputation_method="frequent",
                variables=list(
                    config.pipeline_config.categorical_vars_to_inpute_with_most_freq
                ),
            ),
        ),
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True,
                variables=list(config.pipeline_config.categorical_vars_to_encode),
            ),
        ),
        # Numeric vars
        (
            "cero_replacer",
            ReplaceZeroWithNone(
                variables=list(config.pipeline_config.numerical_vars_to_replace_ceros)
            ),
        ),
        (
            "missing_indicator_numerical",
            AddMissingIndicator(
                variables=list(config.pipeline_config.numerical_vars_with_missing)
            ),
        ),
        (
            "missing_imputation_mean",
            MeanMedianImputer(
                imputation_method="mean",
                variables=list(
                    config.pipeline_config.numerical_vars_with_missing_mean_inputation
                ),
            ),
        ),
        (
            "missing_imputation_median",
            MeanMedianImputer(
                imputation_method="median",
                variables=list(
                    config.pipeline_config.numerical_vars_with_missing_median_inputation
                ),
            ),
        ),
        # Scaling the data
        ("scaler", StandardScaler()),
        # Classification model
        (
            "classification",
            KNeighborsClassifier(
                n_neighbors=config.pipeline_config.knn_n_neighbors,
                weights=config.pipeline_config.knn_weights,
            ),
        ),
    ]
)
