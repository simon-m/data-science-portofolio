import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing as pre
from sklearn.impute import SimpleImputer


@dataclass
class ColumnSubsetTransformer:
    def __init__(self, name: str, pipeline:Pipeline, column_names: list[str]):
        self.column_names = column_names
        self.name = name
        self.pipeline = pipeline


def get_column_transformer(transformers: list[ColumnSubsetTransformer], remainder: str='drop'):
    return ColumnTransformer(
            transformers=[
                (tf.name, tf.pipeline, tf.column_names) for tf in transformers
            ],
        remainder = remainder
    )

def get_linear_model_transformers(X: pd.DataFrame):
    # First attempt with a simple logistic regression as a baseline
    # We therefore need to
    # 1. impute missing values (there's few 3 to 7) so we can keep it simple and use the median / mode
    # 2. standardize numerical values. Again a simple standardization will do
    # 3. one-hot encode categorical values (or dummy if colinearity issues, but shouldn't be the case if we regularize)
    #    The test set doesn't display any unknown values or very small categories so handle_unknown='ignore' will do.
    #    We might need to change that for a real-life deployed model

    numeric_features = list(X.select_dtypes(include=['float64', 'Int64']).columns)
    categorical_features = list(X.select_dtypes(include=['object', 'str']).columns)

    numeric_transform = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='median')),
        ('standardization', pre.StandardScaler())
    ])
    categorical_transform = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='most_frequent')),
        ('onehot', pre.OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer_groups = [
        ColumnSubsetTransformer('numeric', numeric_transform, numeric_features),
        ColumnSubsetTransformer('categorical', categorical_transform, categorical_features)
    ]
    transformer = get_column_transformer(transformer_groups)

    return transformer

def get_tree_model_transformers(X: pd.DataFrame):
    # Simpler pipeline because RF are not impacted by scaling
    categorical_features = list(X.select_dtypes(include=['object', 'str']).columns)

    categorical_transform = Pipeline(steps=[
        # Imputation is needed for OHE
        ('imputation', SimpleImputer(strategy='most_frequent')),
        ('onehot', pre.OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer_groups = [
        ColumnSubsetTransformer('categorical', categorical_transform, categorical_features)
    ]
    transformer = get_column_transformer(transformer_groups, remainder='passthrough')

    return transformer