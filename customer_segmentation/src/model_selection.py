from typing import Any

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import GridSearchCV


def model_selection_cv(X: pd.DataFrame,
                       y: pd.Series,
                       candidates: list[tuple[Any, dict[str, list[Any]|ArrayLike]]],
                       cv_strategy: Any,
                       scoring: str):
    scores = []
    best_score = None
    tuned_classif = None
    for pipeline, param_grid in candidates:
        print(pipeline)
        hp_optim = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring=scoring, verbose=2).fit(X, y)
        if best_score is None or hp_optim.best_score_ > best_score:
            best_score = hp_optim.best_score_
            tuned_classif = hp_optim.best_estimator_
        # Keep all scores to investigate
        scores.append(hp_optim.best_score_)

    return tuned_classif, scores