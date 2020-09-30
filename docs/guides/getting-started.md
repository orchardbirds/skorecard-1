# Getting started guide

A large part of developing scorecards is bucketing the features. `skorecard` provides a range of different [bucketers](/api/bucketers.html) to use.
For example the [AgglomerativeClusteringBucketer](/api/bucketers.html#skorecard.bucketers.bucketers.AgglomerativeClusteringBucketer):

```python
from skorecard import datasets
from skorecard.bucketers import AgglomerativeClusteringBucketer

X, y = datasets.load_uci_credit_card(return_X_y=True)
bucketer = AgglomerativeClusteringBucketer(bins = 10, variables=['LIMIT_BAL'])
bucketer.fit_transform(X)
```

## Scikit-learn pipeline

These bucketers are [scikit-learn](http://scikit-learn.org/) compatible, which means you can use them inside a pipeline:

=== "Basic sklearn pipeline"

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer

    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    bucket_pipeline = make_pipeline(
        EqualWidthBucketer(bins=5, variables=['LIMIT_BAL', 'BILL_AMT1']),
        OrdinalCategoricalBucketer(variables=['EDUCATION', 'MARRIAGE'])
    )

    pipeline = Pipeline([
        ('bucketing', bucket_pipeline),
        ('one-hot-encoding', OneHotEncoder()),
        ('lr', LogisticRegression())
    ])

    pipeline.fit(X, y)
    f"AUC = {roc_auc_score(y, pipeline.predict_proba(X)[:,1]):.4f}"
    ```

=== "With ColumnTransformer"

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    bucket_pipeline = ColumnTransformer([
        ('categorical_preprocessing', OrdinalCategoricalBucketer(), ['EDUCATION', 'MARRIAGE']),
        ('numerical_preprocessing', EqualWidthBucketer(bins=5), ['LIMIT_BAL','BILL_AMT1'])
    ], remainder="passthrough")

    pipeline = Pipeline([
        ('bucketing', bucket_pipeline),
        ('one-hot-encoding', OneHotEncoder()),
        ('lr', LogisticRegression())
    ])

    pipeline.fit(X, y)
    f"AUC = {roc_auc_score(y, pipeline.predict_proba(X)[:,1]):.4f}"
    ```

### Making manual changes to the buckets

You'll often want to incorporate business logic as well.

```python
# Getting the boundaries, TODO
def get_bin_dicts(pipe):

    feature_bucket_mapping = {}

    for trx in pipe.steps:
        for key, value in trx[1].binner_dict_.items():
           feature_bucket_mapping[key] = value

    return feature_bucket_mapping

features_bucket_mapping = get_bin_dicts(pipe)

app = ManualBucketerApp(X, features_bucket_mapping)
# app.run_server(mode="external")
# app.stop_server()
```

Once you have boundaries you like, you save save them as a yml and use them later

```python
# Using the boundaries on new data
transformer = UserInputBucketer(
    binning_dict=features_bucket_mapping, return_object=False, return_boundaries=False)
transformer.fit_transform(df) # returns df
```

https://feature-engine.readthedocs.io/en/latest/discretisers/index.html