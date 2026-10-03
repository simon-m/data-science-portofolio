from sklearn import preprocessing as pre
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, TunedThresholdClassifierCV
from sklearn.metrics import classification_report

df = pd.read_csv("../data/train.csv")
X = df.drop(columns=["Segmentation", "ID"])
y = df["Segmentation"]

# test_df = pd.read_csv("../data/test.csv")
# test_X = train_df.drop(columns=["ID"])
# test_y = train_df["Segmentation"]


print(X.shape, y.shape)

numeric_features = X.select_dtypes(include=['float64', 'Int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# First attempt with a simple logistic regression as a baseline
# We therefore need to
# 1. impute missing values (there's few 3 to 7) so we can keep it simple and use the median / mode
# 2. standardize numerical values. Again a simple standardization will do
# 3. one-hot encode categorical values (or dummy if colinearity issues, but shouldn't be the case if we regularize)
#    The test set doesn't display any unknown values or very small categories so handle_unknown='ignore' will do.
#    We might need to change that for a real-life deployed model

numeric_transform = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='median')),
    ('standardization', pre.StandardScaler())
])
cat_transform = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='most_frequent')),
        ('onehot', pre.OneHotEncoder(handle_unknown='ignore'))
])

feature_gen = ColumnTransformer(
        transformers=[
            ('num', numeric_transform, numeric_features),
            ('cat', cat_transform, categorical_features)
        ],
    remainder = 'drop'
)

# To be tuned, e.g; with LogisticRegressionCV
classifier = LogisticRegression(C=0.5, l1_ratio=0, solver='lbfgs')
pipeline = Pipeline(steps=[('feature_gen', feature_gen),
                               ('classifier', classifier)])

cv_gen = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
cv_res = cross_validate(pipeline, X, y, scoring=['neg_log_loss', 'accuracy', 'balanced_accuracy', 'roc_auc_ovo'], cv=cv_gen, n_jobs=2)


tuned_classif = TunedThresholdClassifierCV(classifier, scoring='balanced_accuracy', cv=cv_gen, n_jobs=2).fit(X, y)

import IPython; IPython.embed()

# pipeline.fit(X, y)
# classif = pipeline.named_steps['classifier']
# classif.predict(X_test)


