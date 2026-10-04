import IPython
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import custom_pipelines
from model_selection import model_selection_cv

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str,  help="Path to the training data")
    args = parser.parse_args()

    df = pd.read_csv(args.train_data)
    X = df.drop(columns=["Segmentation", "ID"])
    y = df["Segmentation"]


    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    log_reg_transform = custom_pipelines.get_linear_model_transformers(X)
    logreg_pipeline = pipeline = Pipeline(steps=[
                                    ('feature_gen', log_reg_transform),
                                    ('classifier', LogisticRegression(C=0.5, l1_ratio=0, solver='lbfgs'))
                                ])
    tree_transform = custom_pipelines.get_tree_model_transformers(X)
    rf_pipeline = line = Pipeline(steps=[
                            ('feature_gen', tree_transform),
                            ('classifier', RandomForestClassifier())
                        ])

    candidates = [
        (
            logreg_pipeline,
            {"classifier__C": np.power(10, np.arange(-4, 5, 0.5))}
        ),
        (
            rf_pipeline,
            {
                "classifier__n_estimators": np.arange(10, 101, 25),
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__criterion": ["gini", "log_loss"],
             }
        )
    ]

    best_model, scores = model_selection_cv(X, y, candidates, cv_strategy, scoring='roc_auc_ovo')

    IPython.embed()

    for name, imp in zip(X.columns, best_model.named_steps["classifier"].feature_importances_):
        print(name, round(100 * imp, 2))

